"""Check shell argument forwarding without loading a GPU framework."""
import json
import os
from pathlib import Path
import subprocess
import pytest

@pytest.mark.parametrize('name',['e2e_dist_eval.sh','e2e_dist_eval_navtest_failures.sh'])
@pytest.mark.parametrize('explicit',[False,True])
def test_collection_and_seed_forwarding(tmp_path,name,explicit):
    scripts=Path(__file__).resolve().parents[2]/'projects/AlgEngine/scripts'
    target=tmp_path/name;target.write_bytes((scripts/name).read_bytes())
    (tmp_path/'e2e_navsim_rescore_utils.sh').write_text('validate_navsim_official_rescore_mode() { :; }\nmaybe_run_navsim_official_rescore() { touch "$RESCORE_CALLED"; }\n')
    fake=tmp_path/'fake_python';fake.write_text('#!/usr/bin/env python3\nimport json,os,sys\nopen(os.environ["ARGS_CAPTURE"],"w").write(json.dumps(sys.argv[1:]))\n');fake.chmod(0o755)
    ckpt=tmp_path/'checkpoint.pth';ckpt.touch()
    env=dict(os.environ,WORLDENGINE_ROOT=str(tmp_path),WORLDENGINE_EVAL_SEED='2',MASTER_PORT='29577',PYTHON_BIN=str(fake),ARGS_CAPTURE=str(tmp_path/'args.json'),RESCORE_CALLED=str(tmp_path/'rescore'))
    env.pop('WORLDENGINE_COLLECT_TMPDIR',None)
    if explicit:env['WORLDENGINE_COLLECT_TMPDIR']=str(tmp_path/'collect with spaces')
    subprocess.run(['bash',str(target),'configs/model.py',str(ckpt),'4'],env=env,check=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    args=json.loads((tmp_path/'args.json').read_text())
    assert args[:4]==['-m','torch.distributed.run','--nproc_per_node=4','--master_port=29577']
    assert args[args.index('--seed')+1]=='2'
    assert args[args.index('--launcher')+1]=='pytorch'
    assert ('--tmpdir' in args)==explicit
    if explicit:assert args[args.index('--tmpdir')+1]==env['WORLDENGINE_COLLECT_TMPDIR']
    assert '' not in args
    assert (tmp_path/'rescore').exists()
