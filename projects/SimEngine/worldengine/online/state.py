"""Lossless snapshots for the explicitly bounded, CPU-only dynamics adapter.

The wire payload is a trusted local pickle, never an untrusted input format.
The structural digest includes hidden state and aliasing, not just ego poses.
"""
from collections import deque
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import io
import pickle
import random
import struct

import numpy as np
from omegaconf import OmegaConf, DictConfig, ListConfig
from shapely.geometry.base import BaseGeometry

from worldengine.base_class.base_runnable import BaseRunnable


def structural_hash(value, static_refs=None, trace=None):
    """Hash supported state by value, retaining container order and object cycles.

    Reject unknown types instead of silently omitting hidden state. Numerical
    arrays and geometries are hashed by value, independent of buffer addresses.
    """
    digest = hashlib.sha256()
    seen = {}

    def emit(tag, data=b''):
        digest.update(tag.encode() + b':' + str(len(data)).encode() + b':' + data)
        if trace is not None:
            trace.append(dict(tag=tag, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                              preview=data[:64].hex() if tag in ('value', 'float', 'geometry')
                              else data[:160].decode('utf-8', errors='replace')))

    def visit(x):
        if static_refs is not None and id(x) in static_refs:
            emit('static_ref', str(static_refs[id(x)]).encode())
        elif x is None:
            emit('none')
        elif isinstance(x, Enum):
            emit('enum', (type(x).__module__ + '.' + type(x).__qualname__).encode())
            visit(x.value)
        elif isinstance(x, (bool, int, str, bytes, float)):
            if isinstance(x, float):
                emit('float', struct.pack('!d', x))
            else:
                emit(type(x).__name__, x if isinstance(x, bytes) else str(x).encode())
        elif isinstance(x, np.generic):
            emit('numpy_scalar', x.dtype.str.encode())
            emit('value', x.tobytes())
        elif isinstance(x, np.ndarray):
            emit('array', x.dtype.str.encode())
            visit(x.shape)
            if x.dtype.hasobject:
                visit(x.tolist())
            else:
                emit('value', np.ascontiguousarray(x).tobytes())
        elif isinstance(x, np.random.RandomState):
            emit('RandomState')
            visit(x.get_state())
        elif isinstance(x, BaseGeometry):
            emit('geometry', x.wkb)
        elif isinstance(x, (DictConfig, ListConfig)):
            emit('omegaconf')
            visit(OmegaConf.to_container(x, resolve=True))
        elif isinstance(x, (set, frozenset)):
            # Sets in map metadata are unordered. Elements cannot contain cycles.
            emit(type(x).__name__)
            for item in sorted(structural_hash(v) for v in x):
                emit('member', item.encode())
        else:
            key = id(x)
            if key in seen:
                emit('ref', str(seen[key][1]).encode())
                return
            # Keep a strong reference: temporary RNG state tuples / x.shape
            # would otherwise be freed and their IDs reused within one walk.
            seen[key] = (x, len(seen))
            emit('class', (type(x).__module__ + '.' + type(x).__qualname__).encode())
            if isinstance(x, dict):
                emit('size', str(len(x)).encode())
                for k, v in x.items():
                    visit(k)
                    visit(v)
            elif isinstance(x, (tuple, list, deque)):
                if isinstance(x, deque):
                    visit(x.maxlen)
                emit('size', str(len(x)).encode())
                for v in x:
                    visit(v)
            elif is_dataclass(x) and type(x).__module__.startswith(('worldengine.', 'nuplan.')):
                for field in fields(x):
                    visit(field.name)
                    visit(getattr(x, field.name))
            elif hasattr(x, '__dict__') and type(x).__module__.startswith(('worldengine.', 'nuplan.')):
                visit(vars(x))
            else:
                raise TypeError('Unreviewed snapshot state type: ' + str(type(x)))
        emit('end')

    visit(value)
    return digest.hexdigest()


@dataclass(frozen=True)
class EngineSnapshot:
    schema: int
    config_hash: str
    static_sha256: str
    state_hash: str
    payload_sha256: str
    payload: bytes


class SnapshotParityError(RuntimeError):
    """Keep replay evidence in memory until the private probe saves the failure."""
    def __init__(self, message, before, expected, actual, bundle, action, details):
        super().__init__(message)
        self.before = before
        self.expected = expected
        self.actual = actual
        self.bundle = bundle
        self.action = action
        self.details = details


MANAGERS = ('scenario_manager', 'map_manager', 'agent_manager')
ENGINE_FIELDS = ('episode_step', 'external_actions', 'global_random_seed',
                 'random_seed', 'np_random', '_episode_start_time')


def validate_engine(engine):
    if tuple(engine.managers) != MANAGERS:
        raise ValueError('Snapshot v1 supports only scenario/map/agent managers; '
                         'renderer, metrics and reward need separate state adapters')
    if not engine.global_config.get('online_deterministic_rng', False):
        raise ValueError('Deterministic agent construction is required for future spawns')
    if engine.global_config.get('visualize_BEV', False):
        raise ValueError('Visualization state is not supported in a dynamics snapshot')
    pending = list(BaseRunnable.__subclasses__())
    while pending:
        cls = pending.pop()
        if 'PARAMETER_SPACE' in cls.__dict__:
            raise ValueError('Subclass parameter space needs an explicit RNG adapter: ' + cls.__name__)
        pending.extend(cls.__subclasses__())


def static_nodes(roots):
    """Retain identities of static graph nodes for pickle persistent references."""
    seen = {}

    def walk(x):
        if x is None or isinstance(x, (str, bytes, bool, int, float, np.generic, Enum)):
            return
        if id(x) in seen:
            return
        seen[id(x)] = x
        if isinstance(x, dict):
            for k, v in x.items():
                walk(k)
                walk(v)
        elif isinstance(x, (tuple, list, deque, set, frozenset)):
            for v in x:
                walk(v)
        elif is_dataclass(x):
            for field in fields(x):
                walk(getattr(x, field.name))
        elif hasattr(x, '__dict__') and type(x).__module__.startswith(('worldengine.', 'nuplan.')):
            walk(vars(x))
        elif isinstance(x, (np.ndarray, np.random.RandomState, BaseGeometry, DictConfig, ListConfig)):
            pass
        else:
            raise TypeError('Unreviewed static state type: ' + str(type(x)))

    walk(roots)
    return list(seen.values())


class SnapshotCodec:
    """One immutable scene/map bundle per episode; small dynamic snapshots.

    Every reference into static geometry uses the same object table, including
    lane references in IDM navigation. Static immutability is audited at group
    boundaries; numeric buffers are also made read-only. Bundle bytes are sent
    once to a spawned worker, never once per candidate.
    """
    def __init__(self, engine=None, bundle=None):
        if (engine is None) == (bundle is None):
            raise ValueError('Provide either a live engine or a trusted static bundle')
        if engine is not None:
            validate_engine(engine)
            roots = (engine.scenario_manager.scenes, engine.current_map)
            nodes = static_nodes(roots)
            self.bundle = pickle.dumps((roots, nodes), protocol=5)
        else:
            self.bundle = bundle
            roots, nodes = pickle.loads(bundle)
        self.roots, self.nodes = roots, nodes  # keep all identities alive
        self.refs = {id(value): index for index, value in enumerate(nodes)}
        self.static_sha256 = hashlib.sha256(self.bundle).hexdigest()
        self.static_hash = structural_hash(roots)
        for node in nodes:
            if isinstance(node, np.ndarray):
                node.flags.writeable = False

    def audit_static(self):
        if structural_hash(self.roots) != self.static_hash:
            raise RuntimeError('A branch mutated static scene/map state; discard group')

    def _dump(self, payload):
        stream = io.BytesIO()
        pickler = pickle.Pickler(stream, protocol=5)
        pickler.persistent_id = lambda obj: ('static', self.refs[id(obj)]) if id(obj) in self.refs else None
        pickler.dump(payload)
        return stream.getvalue()

    def _load(self, encoded):
        def persistent_load(key):
            tag, index = key
            if tag != 'static' or not isinstance(index, int) or not 0 <= index < len(self.nodes):
                raise ValueError('Invalid static reference')
            return self.nodes[index]
        reader = pickle.Unpickler(io.BytesIO(encoded))
        reader.persistent_load = persistent_load
        return reader.load()

    def capture(self, engine):
        validate_engine(engine)
        payload = dict(engine={k: getattr(engine, k) for k in ENGINE_FIELDS},
                       managers=engine.managers, parameter_space=BaseRunnable.PARAMETER_SPACE,
                       numpy_rng=np.random.get_state(), python_rng=random.getstate())
        encoded = self._dump(payload)
        # Roundtrip normalizes the transport representation, including aliases.
        state_hash = structural_hash((self.static_sha256, self._load(encoded)), self.refs)
        return EngineSnapshot(1, structural_hash(engine.global_config), self.static_sha256,
                              state_hash, hashlib.sha256(encoded).hexdigest(), encoded)

    def restore(self, engine, snapshot):
        validate_engine(engine)
        if snapshot.schema != 1 or snapshot.config_hash != structural_hash(engine.global_config):
            raise ValueError('Snapshot schema/config does not match this engine')
        if snapshot.static_sha256 != self.static_sha256:
            raise ValueError('Snapshot belongs to a different scene/map bundle')
        if hashlib.sha256(snapshot.payload).hexdigest() != snapshot.payload_sha256:
            raise ValueError('Snapshot payload checksum mismatch')
        payload = self._load(snapshot.payload)
        if structural_hash((self.static_sha256, payload), self.refs) != snapshot.state_hash:
            raise ValueError('Snapshot structural hash mismatch')
        if tuple(payload['managers']) != MANAGERS:
            raise ValueError('Unexpected managers in snapshot')
        # Replace the graph together; never destroy discarded graphs because
        # destroy() mutates the shared class-level parameter RNG.
        engine._managers = payload['managers']
        for name, manager in engine.managers.items():
            setattr(engine, name, manager)
        for name, value in payload['engine'].items():
            setattr(engine, name, value)
        BaseRunnable.PARAMETER_SPACE = payload['parameter_space']
        np.random.set_state(payload['numpy_rng'])
        random.setstate(payload['python_rng'])

    def mismatch(self, message, before, expected, actual, action):
        """Diagnose a mismatch without tolerances or altering the acceptance gate."""
        left, right = self._load(expected.payload), self._load(actual.payload)
        traces = [[], []]
        for payload, trace in zip((left, right), traces):
            structural_hash((self.static_sha256, payload), self.refs, trace=trace)
        first = next((i for i, (a, b) in enumerate(zip(*traces)) if a != b),
                     min(map(len, traces)))
        components = []

        def check(path, a, b):
            h1, h2 = structural_hash(a, self.refs), structural_hash(b, self.refs)
            if h1 != h2:
                components.append(dict(path=path, expected=h1, actual=h2))

        for key in left:
            check(key, left[key], right[key])
        for key in left['engine']:
            check('engine.' + key, left['engine'][key], right['engine'][key])
        for name in left['managers']:
            a, b = vars(left['managers'][name]), vars(right['managers'][name])
            for key in a.keys() & b.keys():
                check('managers.' + name + '.' + key, a[key], b[key])
        a = left['managers']['agent_manager'].all_agents
        b = right['managers']['agent_manager'].all_agents
        # all_agents only accesses manager dictionaries, not the engine singleton.
        for agent_id in sorted(a.keys() & b.keys()):
            for key in vars(a[agent_id]).keys() & vars(b[agent_id]).keys():
                check('agents.' + agent_id + '.' + key, getattr(a[agent_id], key), getattr(b[agent_id], key))
        details = dict(expected_hash=expected.state_hash, actual_hash=actual.state_hash,
                       expected_payload_sha256=expected.payload_sha256,
                       actual_payload_sha256=actual.payload_sha256,
                       before_hash=before.state_hash, first_trace_difference=first,
                       trace_lengths=list(map(len, traces)), components=components,
                       expected_trace=traces[0][max(0, first-5):first+6],
                       actual_trace=traces[1][max(0, first-5):first+6])
        return SnapshotParityError(message, before, expected, actual, self.bundle, action, details)
