import csv
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'projects/AlgEngine/scripts/diffusiondrive'))
import build_selector_formal_table as table


def dump_csv(path, ids, columns, aggregate=False):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['token','valid',*columns]);writer.writeheader()
        for i,token in enumerate(ids):
            row=dict(token=token,valid='True',**{c:0.5 for c in columns})
            if 'no_at_fault_collisions' in row:
                row['no_at_fault_collisions']=float(i%2==0)
                row['drivable_area_compliance']=float(i%3==0)
            writer.writerow(row)
        if aggregate:writer.writerow(dict(token='overall_average',valid='False',**{c:0 for c in columns}))


@pytest.fixture
def full(tmp_path):
    protocol=tmp_path/'protocol.json'
    nav=[f'n{i}' for i in range(12146)];rare=[f'r{i}' for i in range(288)]
    protocol.write_text(json.dumps(dict(schema_version=1,navtest_tokens=nav,rare_tokens=rare)))
    checkpoint=tmp_path/'checkpoint.pth';checkpoint.write_bytes(b'identity-only-fixture')
    artifacts={}
    for key in table.BLOCKS:
        path=tmp_path/(key+'.csv');ids=nav if key.startswith('openloop_navtest') else rare
        cols=('ade_4s','fde_4s') if key.endswith('_ade') else table.PDM_KEYS
        dump_csv(path,ids,cols,True);artifacts[key]=dict(path=str(path),sha256=table.sha(path))
    ev=dict(status='EVALUATOR_OUTPUTS_COMPLETE',checkpoint=dict(path=str(checkpoint),sha256=table.sha(checkpoint)),
            eval_seed=0,protocol_sha256=table.sha(protocol),artifacts=artifacts)
    manifest=tmp_path/'evaluation.json';manifest.write_text(json.dumps(ev))
    return protocol,manifest,ev


def test_full_exact_coverage_and_joint_success(full):
    protocol,manifest,ev=full
    result=table.build(protocol,manifest,'model','fixture')
    assert result['formal_table_ready']
    assert len(result['row'])==29
    assert result['metrics']['openloop_navtest_ade']['ade_4s']==0.5
    assert result['success_rate']==48/288
    assert result['ignored_aggregate_rows']['closedloop_r']==['overall_average']


@pytest.mark.parametrize('change',['missing','duplicate','wrong_id','nan','invalid','out_of_range'])
def test_bad_scene_rows_rejected(tmp_path,change):
    path=tmp_path/'x.csv';dump_csv(path,['a','b'],table.PDM_KEYS)
    with path.open() as f:rows=list(csv.DictReader(f))
    if change=='missing':rows.pop()
    if change=='duplicate':rows[1]['token']='a'
    if change=='wrong_id':rows[1]['token']='c'
    if change=='nan':rows[0]['score']='nan'
    if change=='invalid':rows[0]['valid']='false'
    if change=='out_of_range':rows[0]['score']='1.2'
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['token','valid',*table.PDM_KEYS]);w.writeheader();w.writerows(rows)
    with pytest.raises(ValueError):table.read_checked(path,['a','b'],table.PDM_KEYS,True)


def test_wrong_identity_rejected(full):
    protocol,manifest,ev=full;ev['checkpoint']['sha256']='0'*64
    manifest.write_text(json.dumps(ev))
    with pytest.raises(ValueError,match='Checkpoint'):table.build(protocol,manifest,'m','n')


def test_diagnostic_protocol_rejected(tmp_path):
    p=tmp_path/'protocol.json';p.write_text(json.dumps(dict(schema_version=1,navtest_tokens=['a'],rare_tokens=['b'])))
    with pytest.raises(ValueError):table.load_protocol(p)
