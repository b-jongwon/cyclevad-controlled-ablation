"""Pinned upstream replay with separate, fixed six-arm extensions. Resume-safe."""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT.parent / 'references/team_cyclevad_20261016'
COMMIT = '7aa58ce893181599c4b5bd6f779fa2461ce3f103'
sys.path.insert(0, str(UPSTREAM / 'src'))
os.environ.setdefault('HF_HOME', str(Path.home() / '.cache/huggingface') if os.name == 'nt' else '/mnt/c/Users/whddn/.cache/huggingface')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
from cycle_vad.data import inventory, write_json, labels_for, frame_paths
from threadpoolctl import threadpool_limits
from addons import read, run_unit, report, PARAMETERS, PRIMARY, ARMS

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def done(path):
    return path.exists() and read(path).get('status') == 'complete'

def prepare(data_root):
    import shutil
    import numpy as np
    import torch
    from cycle_vad.followup import audit_folds
    if subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip() != COMMIT:
        raise RuntimeError('Upstream checkout changed')
    required = {}
    for line in (UPSTREAM / 'results/E0/requirements-lock.txt').read_text().splitlines():
        if '==' in line:
            name, version = line.split('=='); required[name] = version
    mismatches, actual_packages = [], {}
    for name, expected in required.items():
        try: actual = metadata.version(name)
        except metadata.PackageNotFoundError: actual = None
        actual_packages[name] = actual
        if expected != actual:
            mismatches.append(dict(package=name, expected=expected, actual=actual))
    write_json(ROOT / 'environment_mismatches.json', dict(python=platform.python_version(), packages=mismatches,
               blocking=False, scope='User requested identical experimental methods; package/hardware identity not required'))
    assert torch.cuda.is_available() and torch.version.cuda == '12.8'
    if not (ROOT / 'configs').exists():
        shutil.copytree(UPSTREAM / 'configs', ROOT / 'configs')
    # Never silently resume against edited source, settings, or extension code.
    hashes = {str(p.relative_to(UPSTREAM)): sha(p) for base in ['src/cycle_vad', 'configs']
              for p in sorted((UPSTREAM / base).rglob('*')) if p.is_file() and p.suffix in ['.py', '.json']}
    for p in (ROOT / 'configs').rglob('*.json'):
        assert sha(p) == hashes[str(p.relative_to(ROOT))], 'Copied config changed'
    protocol = dict(source_commit=COMMIT, source_hashes=hashes,
                    addon_hashes={p.name: sha(p) for p in [ROOT / 'run.py', ROOT / 'addons.py']},
                    seeds=[42, 43, 44], folds=list(range(5)), scenes=['R01','R02','R03','R04'],
                    primary_baselines=PRIMARY, configurations=ARMS, addon_parameters=PARAMETERS,
                    features='independently extract stride1 and stride2 with original batch8/workers4/FP16 code',
                    calibration='E8B/E12 frame q99; E11B video-equal D-window maximum q99; fusion recalibrated identically',
                    historical_test='fold0 only; strict-v1 unknown mask; source-level event boundaries',
                    environment_policy='Use existing environment; versions/hardware are recorded, not required to match, per user clarification',
                    limitations=['Existing Windows software/hardware differs from upstream; not a bitwise replay',
                                 'Public checkout has no original feature caches or fitted model checkpoints',
                                 'No upstream image content manifest; local image hashes alone cannot prove byte identity',
                                 'Prior tests informed addon design: exploratory, not independent confirmation'],
                    causal_fix='Invalid lag observations excluded from fitting and stall disabled for first4 observations; no future-prefix fill')
    if (ROOT / 'protocol.json').exists():
        assert read(ROOT / 'protocol.json') == protocol, 'Protocol/code changed; use a new output root'
    else:
        write_json(ROOT / 'protocol.json', protocol)
    rows = inventory(data_root, ['R01','R02','R03','R04'])
    fold_audit = audit_folds(read(ROOT / 'configs/experiments/followup/normal_folds.json'), rows)
    expected = read(UPSTREAM / 'results/E0/data_audit.json')
    old = {r['id']: r for r in expected['sequences']}
    assert set(old) == {r['id'] for r in rows}
    for row in rows:
        r = old[row['id']]
        assert row['frames'] == r['frames'], row['id']
        if row['partition'] == 'testing':
            assert sha(row['labels']) == r['label_sha256'], row['id']
            y = labels_for(row)
            assert int((y >= 0).sum()) == r['valid_frames'], row['id']
    write_json(ROOT / 'inventory.json', rows)
    write_json(ROOT / 'dataset_audit.json', dict(status='complete', videos=len(rows),
               upstream_frame_counts_and_label_hashes_match=True, folds=fold_audit,
               upstream_image_content_identity_verified=False))
    env = dict(python=platform.python_version(), platform=platform.platform(),
               packages=actual_packages,
               gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
               gpu_memory=torch.cuda.get_device_properties(0).total_memory,
               source_commit=COMMIT, full_environment_identical=False,
               package_and_python_lock_identical=not mismatches and platform.python_version() == '3.12.3')
    write_json(ROOT / 'environment.json', env)
    (ROOT / 'requirements-actual.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    # Independent baseline replay must retain the original source files byte for byte.
    print(f'PREPARE: {len(rows)} videos; original methods/configs/splits verified; GPU={env["gpu"]}', flush=True)
    return rows

def extract(rows):
    from cycle_vad.features import VisualEncoder, extract_sequence
    import torch
    path = ROOT / 'feature_complete.json'
    if done(path):
        for stride in [1, 2]:
            assert all((ROOT / f'runs/stride{stride}_seed42/cache' / (r['id'] + '.npz')).exists() for r in rows)
        return
    cfg = read(ROOT / 'configs/experiments/replay_pinned.json')
    encoder = VisualEncoder(cfg['encoder'])
    timings = []
    for stride in [2, 1]:
        for row in rows:
            started = time.perf_counter()
            extract_sequence(encoder, row, ROOT / f'runs/stride{stride}_seed42/cache', stride, cfg['batch_size'], cfg['workers'])
            timings.append(dict(id=row['id'], stride=stride, seconds=time.perf_counter()-started))
            write_json(ROOT / 'feature_progress.json', dict(completed=len(timings), total=2*len(rows), last=timings[-1]))
    write_json(path, dict(status='complete', encoder=encoder.identity, timings=timings))
    del encoder
    torch.cuda.empty_cache()

def stats(rows, scene, fold, seed, splits):
    from cycle_vad.followup import run_fold
    from cycle_vad.phase_experiment import run as e8b
    from cycle_vad.process_experiment import run as e9s
    from cycle_vad.paper_experiments import run_e11
    from cycle_vad.paper_subspaces import run_e12
    from cycle_vad.paper_diagnostics import run as e11b
    cfg = read(ROOT / 'configs/experiments/replay_pinned.json')
    cfg['model']['seed'] = seed
    seed_suffix = Path('.') if seed == 42 else Path(f'seed{seed}')
    if seed != 42:
        primary = read(ROOT / 'results/E8' / scene / f'fold{fold}/run.json')
        cfg['model']['harmonics_candidates'] = [primary['diagnostics']['selected_harmonics']]
        cfg['model']['ridge_candidates'] = [primary['diagnostics']['selected_ridge']]
    for stage in ['E8', 'E8B', 'E9S', 'E11', 'E12', 'E11B']:
        suffix = Path(scene) / f'fold{fold}'
        if stage == 'E8':
            dest = ROOT / 'results/E8' / seed_suffix / suffix
        elif stage == 'E9S':
            dest = ROOT / 'results/E9S/stride2' / f'seed{seed}' / suffix
        else:
            dest = ROOT / 'results' / stage / f'seed{seed}' / suffix
        if done(dest / 'run.json'):
            continue
        write_json(ROOT / 'progress.json', dict(status='running', stage=stage, scene=scene, fold=fold, seed=seed))
        print(f'STAGE {stage} seed{seed} {scene}/fold{fold}', flush=True)
        if stage == 'E8':
            run_fold(scene, dict(fold=fold, splits=splits), rows, copy.deepcopy(cfg), ROOT / 'runs/stride2_seed42',
                     ROOT / 'runs/followup' / seed_suffix, ROOT / 'results/E8' / seed_suffix)
        elif stage == 'E8B': e8b(ROOT, rows, scene, fold, seed, splits)
        elif stage == 'E9S': e9s(ROOT, rows, scene, fold, seed, splits, stride=2)
        elif stage == 'E11': run_e11(ROOT, rows, scene, fold, seed, splits)
        elif stage == 'E12': run_e12(ROOT, rows, scene, fold, seed, splits)
        else: e11b(ROOT, rows, scene, fold, seed, splits)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, default=Path('D:/종프 학습/IPAD_dataset') if os.name == 'nt' else Path('/mnt/d/종프 학습/IPAD_dataset'))
    p.add_argument('--stage', choices=['all','prepare','extract','stats','addons','report'], default='all')
    p.add_argument('--seeds', type=int, nargs='+', default=[42,43,44])
    p.add_argument('--folds', type=int, nargs='+', default=list(range(5)))
    p.add_argument('--scenes', nargs='+', default=['R01','R02','R03','R04'])
    a = p.parse_args()
    if a.stage == 'report':
        report(ROOT); return
    began = time.perf_counter()
    rows = prepare(a.data_root)
    with threadpool_limits(limits=4):
        if a.stage in ['all','extract']: extract(rows)
        if a.stage in ['all','stats','addons']:
            assert done(ROOT / 'feature_complete.json'), 'Run extraction first'
            assignments = read(ROOT / 'configs/experiments/followup/normal_folds.json')
            for seed in a.seeds:
                for fold in a.folds:
                    for scene in a.scenes:
                        splits = assignments['scenes'][scene][fold]['splits']
                        if a.stage in ['all','stats']: stats(rows, scene, fold, seed, splits)
                        if a.stage in ['all','addons']: run_unit(ROOT, rows, scene, fold, seed, splits)
                        report(ROOT)
    write_json(ROOT / f'{a.stage}_completion.json', dict(status='complete', seeds=a.seeds, folds=a.folds,
               scenes=a.scenes, wall_seconds=time.perf_counter()-began))
    write_json(ROOT / 'progress.json', dict(status='complete', stage=a.stage, seeds=a.seeds, folds=a.folds, scenes=a.scenes))
    print('COMPLETE', a.stage, flush=True)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        write_json(ROOT / 'failure.json', dict(status='failed', error=str(e), traceback=traceback.format_exc()))
        raise
