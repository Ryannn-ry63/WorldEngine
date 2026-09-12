"""Decision-feedback uses the audited deployment transport, never simulator labels."""
import os
_base_ = ['./e2e_diffusiondrive_selector_feedback.py']
if os.environ.get('DIFFUSIONDRIVE_FEEDBACK_RESEARCH_METHOD') != 'selector_decision_feedback_v1':
    raise RuntimeError('Explicit decision-feedback research identity required')
selector_rollout_contract = dict(research_method='selector_decision_feedback_v1')
