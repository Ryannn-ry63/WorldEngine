"""Bounded inherited memfd image transport; JSON control, no frame files/pickle.

One writer and one reader, request/reply ownership. The writer may reuse the
buffer only after the reader has copied/verified the preceding descriptor.
An inherited anonymous memfd avoids cross-interpreter resource_tracker unlink.
"""
import hashlib
import mmap
import os
import tempfile
import numpy as np

CAPACITY = 32 * 1024 * 1024


def metadata_to_wire(value):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError('Object arrays are not observation metadata')
        return {'__ndarray__': True, 'dtype': value.dtype.str,
                'shape': list(value.shape), 'data': value.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise TypeError('Observation keys must be strings')
        return {'__mapping__': [[k, metadata_to_wire(v)] for k, v in value.items()]}
    if isinstance(value, (list, tuple)):
        return [metadata_to_wire(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError('Unsupported observation metadata: ' + str(type(value)))


def metadata_from_wire(value):
    if isinstance(value, dict):
        if value.get('__ndarray__') is True:
            dtype = np.dtype(value['dtype'])
            if dtype.hasobject or dtype.kind not in 'biufUS':
                raise ValueError('Unsupported metadata dtype')
            arr = np.asarray(value['data'], dtype=dtype)
            if list(arr.shape) != value['shape']:
                raise ValueError('Metadata array shape mismatch')
            return arr
        if set(value) == {'__mapping__'}:
            pairs = value['__mapping__']
            if any(not isinstance(k, str) for k, _ in pairs) or len({k for k, _ in pairs}) != len(pairs):
                raise ValueError('Invalid metadata keys')
            return {k: metadata_from_wire(v) for k, v in pairs}
        raise ValueError('Unknown metadata mapping tag')
    if isinstance(value, list):
        return [metadata_from_wire(v) for v in value]
    return value


class ImageBuffer:
    def __init__(self, fd=None):
        self._backing = None
        if fd is None:
            flags = getattr(os, 'O_TMPFILE', 0) | os.O_RDWR
            try:
                self.fd = os.open('/dev/shm', flags, 0o600)
            except (AttributeError, OSError):
                # Anonymous TemporaryFile is still inherited by descriptor and
                # never exposed as a named frame file.
                self._backing = tempfile.TemporaryFile()
                self.fd = self._backing.fileno()
            os.ftruncate(self.fd, CAPACITY)
        else:
            self.fd = fd
        if os.fstat(self.fd).st_size != CAPACITY:
            raise ValueError('Unexpected shared camera buffer size')
        self.memory = mmap.mmap(self.fd, CAPACITY)
        self.pending = None

    def publish(self, images, identity):
        if self.pending is not None:
            raise RuntimeError('Camera buffer still belongs to previous observation')
        if not images or sum(len(v) for v in images.values()) > CAPACITY:
            raise ValueError('Camera payload empty or larger than shared buffer')
        entries, offset = [], 0
        for name, blob in images.items():
            if not isinstance(name, str) or not isinstance(blob, bytes) or not blob:
                raise TypeError('Expected named JPEG byte strings')
            self.memory[offset:offset+len(blob)] = blob
            entries.append(dict(name=name, offset=offset, size=len(blob),
                                sha256=hashlib.sha256(blob).hexdigest()))
            offset += len(blob)
        self.pending = dict(identity)
        return dict(schema_version=1, identity=dict(identity), images=entries, total=offset)

    def receive(self, descriptor, identity):
        if descriptor['schema_version'] != 1 or descriptor['identity'] != identity:
            raise ValueError('Stale camera descriptor')
        output, offset = {}, 0
        for item in descriptor['images']:
            size = item['size']
            if not isinstance(size, int) or size < 1 or item['offset'] != offset or offset+size > CAPACITY:
                raise ValueError('Invalid camera buffer bounds')
            if item['name'] in output:
                raise ValueError('Duplicate camera name')
            blob = self.memory[offset:offset+size]
            if hashlib.sha256(blob).hexdigest() != item['sha256']:
                raise ValueError('Camera payload checksum mismatch')
            output[item['name']] = blob
            offset += size
        if not output or offset != descriptor['total']:
            raise ValueError('Camera descriptor length mismatch')
        self.pending = dict(identity)
        return output

    def release(self, identity):
        if self.pending != identity:
            raise ValueError('Camera acknowledgement identity mismatch')
        self.pending = None

    def close(self):
        self.memory.close()
        if self._backing is not None:
            self._backing.close()
        else:
            os.close(self.fd)
