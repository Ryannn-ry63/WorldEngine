"""Training interventions and unmodified NR/R evaluation share one frozen forward."""
import os
_base_ = ['./e2e_diffusiondrive_selector_cfpi_deployment.py']
_mode = os.environ['DIFFUSIONDRIVE_FEEDBACK_REACT_TYPE']
_cohort = os.environ['DIFFUSIONDRIVE_FEEDBACK_COHORT']
if _mode not in ('NR','R') or _cohort not in ('train','development','common'):
    raise RuntimeError('Explicit mode and cohort required')
selector_rollout_contract = dict(
    research_method='selector_feedback_repair_v2',react_type=_mode,source_data_split=_cohort,
    development_consumed=_cohort=='development',test_consumed=False,
    legacy_exposed_benchmark=True,independent_unseen_test=False)
