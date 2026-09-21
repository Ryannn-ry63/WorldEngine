#!/usr/bin/env python3
"""Bind explicit completed evaluator files to a checkpoint and frozen protocol.
This records provenance supplied by the operator; it is not a coverage or runtime check.
"""
import argparse,json
from pathlib import Path
from build_selector_formal_table import BLOCKS,sha,load_protocol

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--eval-seed',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    for key in BLOCKS:p.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    a=p.parse_args();load_protocol(a.protocol)
    def artifact(path):return dict(path=str(path.resolve(strict=True)),sha256=sha(path))
    result=dict(status='EVALUATOR_OUTPUTS_COMPLETE',checkpoint=artifact(a.checkpoint),
                protocol_sha256=sha(a.protocol),eval_seed=a.eval_seed,
                artifacts={key:artifact(getattr(a,key)) for key in BLOCKS},
                note='Explicit operator attribution; final table independently checks exact coverage.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
if __name__=='__main__':main()
