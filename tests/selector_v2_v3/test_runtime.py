import json
import os
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'projects/AlgEngine/scripts/diffusiondrive'))
import selector_runtime as rt


def config(tmp_path):
    source=rt.CODE/'selector.example.json'
    p=tmp_path/'settings.json';p.write_text(source.read_text());return p


def test_explicit_paths_and_no_old_pythonpath(tmp_path,monkeypatch):
    monkeypatch.setenv('PYTHONPATH','/old/project')
    monkeypatch.setenv('WORLDENGINE_ROOT','/old/project')
    monkeypatch.setenv('DIFFUSIONDRIVE_GRPO_KL_WEIGHT','0.9')
    cfg,env=rt.environment(config(tmp_path),'0,2')
    assert '/old/project' not in env['PYTHONPATH']
    assert env['WORLDENGINE_ROOT']==str(rt.CODE)
    assert 'DIFFUSIONDRIVE_GRPO_KL_WEIGHT' not in env
    assert env['NAVSIM_OFFICIAL_RESCORE']=='always'
    assert env['CUDA_VISIBLE_DEVICES']=='0,2'


def test_duplicate_devices_rejected(tmp_path):
    with pytest.raises(ValueError):rt.environment(config(tmp_path),'0,0')


def test_dry_run_does_not_create_output(tmp_path):
    out=tmp_path/'output'
    rt.run_commands([(['/bin/true'],tmp_path,'test.log')],out,os.environ.copy(),1,dry=True)
    assert not out.exists()


def test_failure_has_persistent_status(tmp_path):
    out=tmp_path/'failed'
    with pytest.raises(RuntimeError):
        rt.run_commands([([sys.executable,'-c','raise SystemExit(2)'],tmp_path,'test.log')],out,os.environ.copy(),3)
    assert json.loads((out/'process_status.json').read_text())['status']=='FAILED'
    assert (out/'test.log').is_file()
