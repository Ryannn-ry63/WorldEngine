"""Independent file-bridge oracle, used only by bounded visual acceptance probes."""
import copy
from pathlib import Path
import pickle
import tempfile
import numpy as np
import torch
from mmcv.parallel import DataContainer, collate


def assert_same(a, b, path='input'):
    if isinstance(a, DataContainer):
        if not isinstance(b, DataContainer):
            raise AssertionError(path + ': container type')
        for key in ('stack', 'cpu_only', 'padding_value', 'pad_dims'):
            if getattr(a, key) != getattr(b, key):
                raise AssertionError(path + ': container ' + key)
        return assert_same(a.data, b.data, path + '.data')
    if torch.is_tensor(a):
        if not torch.is_tensor(b) or a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            raise AssertionError(path + ': tensor mismatch')
    elif isinstance(a, np.ndarray):
        if not isinstance(b, np.ndarray) or a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a,b):
            raise AssertionError(path + ': array mismatch')
    elif isinstance(a, dict):
        if not isinstance(b, dict) or list(a) != list(b):
            raise AssertionError(path + ': dictionary keys/order')
        for k in a: assert_same(a[k], b[k], path + '.' + str(k))
    elif isinstance(a, (list, tuple)):
        if type(a) != type(b) or len(a) != len(b):
            raise AssertionError(path + ': sequence mismatch')
        for i, (x,y) in enumerate(zip(a,b)): assert_same(x,y,path+'.'+str(i))
    elif type(a) != type(b) or a != b:
        raise AssertionError(path + ': scalar mismatch')


def file_oracle(dataset_cfg, frames, cameras):
    from mmdet3d_plugin.datasets.navsim_openscene_closed_loop import NavSimOpenSceneE2EClosedLoop
    # These temporary files are solely the old-pipeline acceptance oracle, not
    # IPC. The actual live path never writes frames or rebuilds a dataset.
    with tempfile.TemporaryDirectory(prefix='i3-file-oracle-') as directory:
        root = Path(directory)
        for frame, view in zip(frames, cameras):
            for name, camera in frame['cams'].items():
                relative = Path(camera['data_path'])
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError('Unsafe oracle camera path')
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as stream: stream.write(view[name])
        path = root/'window.pkl'
        with path.open('xb') as stream: pickle.dump(dict(infos=copy.deepcopy(frames)), stream)
        cfg = copy.deepcopy(dict(dataset_cfg)); cfg.pop('type', None)
        cfg.update(ann_file=str(path), metric_cache_path=None, test_mode=True)
        cfg['pipeline'][0]['img_root'] = str(root)
        dataset = NavSimOpenSceneE2EClosedLoop(**cfg)
        return collate([dataset[0]], samples_per_gpu=1)


def result_parity(memory, file):
    if memory['token'] != file['token'] or memory['chosen_ind'] != file['chosen_ind']:
        raise AssertionError('File/memory token or selected action differs')
    a,b = memory['diffusiondrive_rollout_context'],file['diffusiondrive_rollout_context']
    if set(a) != set(b): raise AssertionError('File/memory context fields differ')
    errors = {}
    for key in a:
        if key == 'schema_version':
            if a[key] != b[key]: raise AssertionError('Context schema differs')
            continue
        x,y = np.asarray(a[key]),np.asarray(b[key])
        if x.shape != y.shape or x.dtype != y.dtype or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise AssertionError('Invalid parity output ' + key)
        errors[key] = float(np.max(np.abs(x.astype(float)-y.astype(float))))
        if not np.allclose(x,y,atol=1e-5,rtol=1e-4):
            raise AssertionError('File/memory forward differs: ' + key + ' error=' + str(errors[key]))
    if not np.allclose(memory['trajectory'],file['trajectory'],atol=1e-5,rtol=1e-4):
        raise AssertionError('File/memory deployed trajectory differs')
    return errors
