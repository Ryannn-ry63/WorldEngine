"""Continuous online-session engineering probe.

The rebuild reference starts one bounded visual probe per episode, while the
learner checkpoint, optimizer, policy version and private RNG continue across
episode boundaries.  It is deliberately separate from the accepted 8-step
probe and is not formal training or an effect evaluation.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import contextlib
import sys
import time

from .paths import checked_path
from .scene_manifest import load_assignments
from .session_lifecycle import SessionLedger


def _episode_settings(base, row, path, episode):
    cfg = dict(base)
    # Keep the manifest's audited assignment while deriving a deterministic,
    # disjoint RNG namespace for each repeated episode.
    stride = 1000003
    cfg.update(
        visual_scene_id=row['scene_id'],
        visual_scene_path=row['scene_path'],
        online_candidate_seed=(row['candidate_seed'] + episode * stride) % (2**31),
        online_scene_seed=(row['scene_seed'] + episode * stride) % (2**31),
        online_action_seed=(row['candidate_seed'] + episode * stride + 17) % (2**31),
    )
    path.write_text(json.dumps(cfg, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scene-manifest', type=Path)
    parser.add_argument('--episodes', type=int, default=16)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mode', choices=('rebuild', 'resident'), default='rebuild')
    parser.add_argument('--throughput', action='store_true')
    parser.add_argument('--branch-workers', type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.episodes <= 16:
        parser.error('--episodes must be in [1, 16]')
    if not 1 <= args.steps <= 8 or args.seed < 0:
        parser.error('Use steps 1..8 and a nonnegative seed')
    if not 0 <= args.branch_workers <= 20:
        parser.error('--branch-workers must be in [0, 20]')
    settings = checked_path(args.settings)
    output = checked_path(args.output.absolute(), must_exist=False)
    if output.exists():
        raise FileExistsError('Use a fresh output basename: ' + str(output))
    code = Path(__file__).resolve().parents[5]
    if code == output or code in output.parents:
        raise ValueError('Session output must be outside code')
    base = json.loads(settings.read_text())
    if args.scene_manifest:
        manifest, rows = load_assignments(args.scene_manifest, settings, 2, verify_files=False)
        # A single-rank session uses the first explicitly audited assignment;
        # multi-rank orchestration supplies one settings file per rank.
        row = rows[0]
        assignment = dict(scene_manifest=str(args.scene_manifest.resolve()),
                           manifest_schema=manifest['schema'], scene_id=row['scene_id'])
    else:
        required = ('visual_scene_id', 'visual_scene_path')
        if any(not base.get(k) for k in required):
            raise ValueError('Session requires visual_scene_id and visual_scene_path, or --scene-manifest')
        row = dict(scene_id=base['visual_scene_id'], scene_path=base['visual_scene_path'],
                   candidate_seed=base.get('online_candidate_seed', args.seed),
                   scene_seed=base.get('online_scene_seed', args.seed))
        assignment = dict(scene_id=row['scene_id'])
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = ('ONLINE_SESSION_RESIDENT_PARENT' if args.mode == 'resident'
             else 'ONLINE_SESSION_REBUILD_REFERENCE')
    summary = dict(status='RUNNING', stage=stage, mode=args.mode,
                   episodes_requested=args.episodes, steps_per_episode=args.steps,
                   decision_opportunities=args.episodes * args.steps, assignment=assignment,
                   episodes=[], actual_optimizer_steps=0, update_attempts=0)
    ledger = SessionLedger(output.stem, policy_version=0)
    started = time.monotonic()
    previous_checkpoint = None
    previous_version = 0
    try:
        for episode in range(args.episodes):
            token = ledger.start_episode(row['scene_id'],
                                         (row['scene_seed'] + episode * 1000003) % (2**31))
            episode_output = output.with_name(output.stem + '.episode%02d.json' % episode)
            episode_settings = output.with_name(output.stem + '.episode%02d.settings.json' % episode)
            episode_log = output.with_name(output.stem + '.episode%02d.log' % episode)
            _episode_settings(base, row, episode_settings, episode)
            command = [base['algengine_python'], '-u', '-m', 'innovation3.online_probe',
                       '--settings', str(episode_settings), '--output', str(episode_output),
                       '--steps', str(args.steps), '--seed', str(args.seed),
                       '--policy-version', str(previous_version)]
            if args.throughput:
                command.append('--throughput')
            if args.branch_workers:
                command += ['--branch-workers', str(args.branch_workers)]
            if args.mode == 'resident':
                command.append('--resident')
            elif previous_checkpoint:
                command += ['--resume', str(previous_checkpoint)]
            launch = time.monotonic()
            if args.mode == 'resident':
                # Keep the model/learner process alive. The visual worker is
                # still rebuilt per episode until its persistent channel is
                # added; this mode isolates the parent startup saving first.
                from . import online_probe
                old_argv = sys.argv
                with episode_log.open('x') as log, contextlib.redirect_stdout(log), \
                        contextlib.redirect_stderr(log):
                    sys.argv = ['innovation3.online_probe'] + command[4:]
                    try:
                        return_code = online_probe.main()
                    finally:
                        sys.argv = old_argv
                process = type('Result', (), {'returncode': return_code})()
            else:
                with episode_log.open('x') as log:
                    process = subprocess.run(command, env=os.environ.copy(), stdout=log,
                                             stderr=subprocess.STDOUT, check=False)
            if process.returncode not in (0, 2):
                raise RuntimeError('Episode %d failed with code %d; log=%s' %
                                   (episode, process.returncode, episode_log))
            report_path = episode_output
            if not report_path.exists():
                raise RuntimeError('Episode %d did not produce a report' % episode)
            report = json.loads(report_path.read_text())
            if report.get('status') != 'PASS_STRICT_ONLINE_THROUGHPUT_PROBE' and args.throughput:
                raise RuntimeError('Episode %d did not pass strict throughput contract' % episode)
            initial = report.get('policy_version_initial', 0)
            if initial != previous_version or report.get('update_attempts') != args.steps:
                raise RuntimeError('Episode %d learner boundary mismatch' % episode)
            version = report.get('policy_version')
            if type(version) is not int or version < previous_version:
                raise RuntimeError('Episode %d policy version regressed' % episode)
            for event in report.get('events', []):
                if event.get('kind') != 'online_update':
                    continue
                ledger.record_decision(token, event['version_before'],
                                       event['version_after'], bool(event['optimized']))
            receipt = ledger.close_episode(args.steps)
            checkpoint = episode_output.with_suffix('.online.pt')
            if not checkpoint.exists():
                raise RuntimeError('Episode %d missing learner checkpoint' % episode)
            summary['episodes'].append(dict(episode=episode, scene=row['scene_id'],
                status=report.get('status'), policy_version_before=previous_version,
                policy_version_after=version, update_attempts=report.get('update_attempts'),
                actual_optimizer_steps=report.get('actual_optimizer_steps'),
                elapsed_seconds=report.get('elapsed_seconds'), launch_seconds=time.monotonic()-launch,
                warmup_seconds=report.get('throughput', {}).get('warmup_seconds'),
                ledger=receipt,
                report=str(report_path), checkpoint=str(checkpoint), log=str(episode_log)))
            previous_checkpoint, previous_version = checkpoint, version
        summary.update(status=('PASS_ONLINE_SESSION_RESIDENT_PARENT' if args.mode == 'resident'
                               else 'PASS_ONLINE_SESSION_REBUILD_REFERENCE'),
                       actual_optimizer_steps=previous_version,
                       update_attempts=args.episodes * args.steps,
                       elapsed_seconds=time.monotonic() - started,
                       final_checkpoint=str(previous_checkpoint),
                       final_policy_version=previous_version,
                       ledger_boundary=ledger.boundary(),
                       formal_ready=False, used_for_formal_training=False)
    except BaseException as error:
        summary.update(status=('FAIL_ONLINE_SESSION_RESIDENT_PARENT' if args.mode == 'resident'
                               else 'FAIL_ONLINE_SESSION_REBUILD_REFERENCE'), error=repr(error),
                       elapsed_seconds=time.monotonic() - started)
        raise
    finally:
        output.write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(dict(status=summary['status'], report=str(output))), flush=True)
    return 0 if summary['status'].startswith('PASS_ONLINE_SESSION_') else 2


if __name__ == '__main__':
    raise SystemExit(main())
