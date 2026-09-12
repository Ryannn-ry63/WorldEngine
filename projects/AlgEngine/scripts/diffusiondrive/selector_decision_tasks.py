"""Subprocess boundaries for CPU preparation/reporting during charged allocation."""
import argparse
from pathlib import Path
import torch
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_decision_common as x
import selector_decision_data as data
import selector_decision_collections as q
from report_selector_decision import report, name


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('diagnostic_bundle','diagnostic_report','sentinel',
                                    'query_bundle','query_cache','evaluation_bundle','report','cached_preflight'))
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--arm',choices=('P2','P3'))
    p.add_argument('--seed',type=int,choices=x.SEEDS)
    p.add_argument('--step',type=int,choices=x.QUERY_STEPS)
    p.add_argument('--generation',type=int,choices=(1,2))
    p.add_argument('--entry',type=Path)
    p.add_argument('--reference',type=Path)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    a = p.parse_args()
    torch.set_num_threads(4)
    if a.action == 'cached_preflight':
        data.preflight(a.run_root,a.device)
    elif a.action == 'diagnostic_bundle':
        q.diagnostic_bundle(a.run_root)
    elif a.action == 'diagnostic_report':
        q.diagnostic_report(a.run_root)
    elif a.action == 'sentinel':
        q.compare_sentinel(dict(sentinel=d.artifact(a.entry),reference=d.artifact(a.reference)))
    elif a.action == 'query_bundle':
        q.query_bundle(a.run_root,a.arm,a.seed,a.step)
    elif a.action == 'query_cache':
        q.build_query_cache(a.run_root,a.arm,a.seed,a.generation)
    elif a.action == 'evaluation_bundle':
        q.require_pilot(a.run_root)
        entries = [q.make(a.run_root,name(cohort,mode,policy),cohort,mode,policy)
                   for cohort,mode,policy in x.evaluation_conditions()]
        c.locked_json(a.run_root/'pilot_collections.json',dict(status='PASS',collections=entries))
    else:
        result = report(a.run_root)
        print({k:result[k] for k in ('status','decision')},flush=True)
        if result['status'] != 'PASS':
            raise SystemExit(2)


if __name__ == '__main__':
    main()
