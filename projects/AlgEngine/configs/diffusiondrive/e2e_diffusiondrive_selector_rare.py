"""Same deployment transport, with truthful rare-evaluation exposure metadata."""
import os

_base_ = ['./e2e_diffusiondrive_selector_cfpi_deployment.py']

rare_cohort = os.environ.get('DIFFUSIONDRIVE_RARE_EVAL_COHORT', '')
if rare_cohort not in ('bridge', 'development', 'common', 'confirmation'):
    raise RuntimeError('Rare deployment requires an explicit frozen evaluation cohort')

selector_rollout_contract = dict(
    research_method='selector_rare_retention_v1',
    source_data_split=rare_cohort,
    development_consumed=(rare_cohort == 'development'),
    test_consumed=(rare_cohort == 'confirmation'),
    legacy_exposed_benchmark=True,
    independent_unseen_test=False,
    evaluation_role='frozen_candidate_confirmation' if rare_cohort == 'confirmation' else 'research_screening')
