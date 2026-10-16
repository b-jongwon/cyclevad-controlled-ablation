"""Six fixed ablations. Inference accepts features/scores, never annotations."""
from pathlib import Path
import json
import joblib
import numpy as np
from cycle_vad.data import hold, load_cache, labels_for, write_json
from cycle_vad.model import descriptor
from cycle_vad.metrics import frame_metrics, event_metrics
from cycle_vad.paper_experiments import calibrate

ARMS = ['baseline', 'adaptive_only', 'stall_alarm_only', 'stall_fusion_only',
        'adaptive_fusion', 'adaptive_fusion_stall_alarm']
TITLES = ['Baseline', '+ Adaptive threshold only', '+ Stall alarm only',
          '+ Stall score fusion only', '+ Adaptive threshold + Fusion',
          '+ Adaptive threshold + Fusion + Stall alarm']
PRIMARY = {'E8B': 'M3__conditional_only', 'E12': 'S1_f100__head_only', 'E11B': 'CP'}
PARAMETERS = dict(phase_bins=16, minimum_bin_samples=64, lags=[1, 4], ema=.5,
                  stall_quantile=.995, consecutive=3, warmup=24, low_fraction=.2,
                  gain=.3, stall_multiplier=1.1)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def motion(g, lag):
    # Preserve the prior float32 cosine calculation; remove future-prefix fill.
    g = np.asarray(g, dtype=np.float32)
    out = np.full(len(g), np.nan)
    if len(g) > lag:
        out[lag:] = 1. - np.sum(g[lag:] * g[:-lag], axis=1)
    return out

class LowerTail:
    def fit(self, values, theta):
        valid = np.isfinite(values)
        values, theta = values[valid], theta[valid]
        if not len(values):
            raise ValueError('Empty normal motion reference')
        fallback = np.sort(values)
        bins = np.floor((theta % 1) * 16).astype(int).clip(0, 15)
        self.refs = [np.sort(values[bins == b]) if (bins == b).sum() >= 64 else fallback for b in range(16)]
        return self

    def score(self, values, theta):
        out = np.zeros(len(values))
        bins = np.floor((theta % 1) * 16).astype(int).clip(0, 15)
        for b, ref in enumerate(self.refs):
            mask = (bins == b) & np.isfinite(values)
            out[mask] = -np.log((np.searchsorted(ref, values[mask], side='right') + 1) / (len(ref) + 1))
        return out

def stall_score(g, theta, tails):
    raw = np.minimum.reduce([tail.score(motion(g, lag), theta) for lag, tail in tails.items()])
    out = np.zeros(len(g))
    for j in range(4, len(g)):
        out[j] = .5 * raw[j] + .5 * out[j - 1]
    return out

def low(values):
    values = np.asarray(values)[:24]
    return float(np.median(np.sort(values)[:max(1, int(np.ceil(.2 * len(values))))]))

def consecutive(values):
    out = np.zeros(len(values), dtype=bool)
    count = 0
    for j, value in enumerate(values):
        count = count + 1 if value else 0
        out[j] = count >= 3
    return out

def predict(base, stall, fitted):
    base, stall = np.asarray(base), np.asarray(stall)
    fusion = np.maximum(base / max(fitted['base_threshold'], 1e-6),
                        1.1 * stall / max(fitted['stall_threshold'], 1e-6))
    scores, alarms, thresholds = [], [], []
    st_alarm = consecutive(stall > fitted['stall_threshold'])
    for arm in ARMS:
        fused = 'fusion' in arm
        score = fusion if fused else base
        threshold = np.full(len(score), fitted['fusion_threshold'] if fused else fitted['base_threshold'])
        if 'adaptive' in arm and len(score) >= 24:
            reference = fitted['fusion_low'] if fused else fitted['base_low']
            threshold[23:] += .3 * max(0., low(score) - reference)
        alarm = score > threshold
        if arm in ['stall_alarm_only', 'adaptive_fusion_stall_alarm']:
            alarm = alarm | st_alarm
        scores.append(score); alarms.append(alarm); thresholds.append(threshold)
    return np.column_stack(scores), np.column_stack(alarms), np.column_stack(thresholds)

def measure(y, score, alarm):
    m = frame_metrics(y, score, 0.)
    m.pop('threshold')
    valid = y >= 0
    yy, aa = y[valid], alarm[valid]
    tp, fp = int(np.sum(aa & (yy == 1))), int(np.sum(aa & (yy == 0)))
    fn, tn = int(np.sum(~aa & (yy == 1))), int(np.sum(~aa & (yy == 0)))
    m.update(tp=tp, fp=fp, fn=fn, tn=tn,
             fpr=fp / (fp + tn) if fp + tn else None,
             recall=tp / (tp + fn) if tp + fn else None,
             precision=tp / (tp + fp) if tp + fp else 0.,
             f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)
    # Called per source only; event intervals never cross video boundaries.
    return m

def operating_threshold(streams, splits, experiment, period, deadline):
    if experiment == 'E11B':
        return float(calibrate({i: s[:, None] for i, s in streams.items()}, splits, period, deadline)[0][3, 0])
    return float(np.quantile(np.concatenate([streams[i] for i in splits['threshold']]), .99, method='higher'))

def get_bank(root, experiment, seed, scene, fold, splits, rows):
    suffix = Path(f'seed{seed}') / scene / f'fold{fold}'
    meta = read(root / 'results' / experiment / suffix / 'run.json')
    groups = {k: list(v) for k, v in splits.items() if k not in ['fit', 'validation']}
    if fold == 0:
        groups['historical_test'] = [r['id'] for r in rows if r['scene'] == scene and r['partition'] == 'testing']
    ids = sum(groups.values(), [])
    bank, limits = {}, {}
    if experiment == 'E8B':
        for m in meta['models']:
            for r, threshold in m['thresholds'].items():
                limits[m['arm'] + '__' + r] = threshold
    elif experiment == 'E12':
        for m in meta['models']:
            if m['status'] != 'complete':
                continue
            for j, readout in enumerate(['head_only', 'appearance_head']):
                limits[f"{m['arm']}_f{int(100*m['fraction'])}__{readout}"] = m['frame_thresholds'][j]
    else:
        limits = dict(zip(meta['arms'], meta['window_thresholds'][3]))
    for i in ids:
        _, part, seq = i.split('/')
        folder = root / 'runs' / experiment / suffix
        path = folder / ('scores' if experiment == 'E8B' else '') / f'{part}_{seq}.npz'
        with np.load(path, allow_pickle=False) as data:
            if experiment == 'E8B':
                bank[i] = {k: data[k].copy() for k in limits}
            elif experiment == 'E12':
                bank[i] = {key + '__' + readout: data[key][:, j].copy()
                           for key in data.files if key != 'labels'
                           for j, readout in enumerate(['head_only', 'appearance_head'])}
            else:
                bank[i] = {k: data['scores'][:, j].copy() for j, k in enumerate(meta['arms'])}
    return bank, limits, groups, meta

def run_unit(root, rows, scene, fold, seed, splits):
    lookup = {r['id']: r for r in rows}
    suffix = Path(f'seed{seed}') / scene / f'fold{fold}'
    for experiment in ['E8B', 'E12', 'E11B']:
        dest = root / 'ablation' / experiment / suffix
        if (dest / 'complete.json').exists():
            continue
        bank, limits, groups, meta = get_bank(root, experiment, seed, scene, fold, splits, rows)
        ids = list(bank)
        global_bank, angles = {}, {}
        if experiment == 'E8B':
            seed_suffix = Path('.') if seed == 42 else Path(f'seed{seed}')
            tracker = joblib.load(root / 'runs/followup' / seed_suffix / scene / f'fold{fold}/train2.joblib').tracker
        for i in ids:
            data = load_cache(root / f"runs/stride{2 if experiment == 'E8B' else 1}_seed42/cache" / (i + '.npz'))
            global_bank[i] = data['global'] if experiment == 'E8B' else data['global'][::2]
            if experiment == 'E8B':
                angles[i] = tracker.predict(descriptor(data), data['indices'])['angle']
            else:
                _, part, seq = i.split('/')
                with np.load(root / 'runs/E15A' / suffix / f'{part}_{seq}.npz') as d:
                    angles[i] = d['theta'][::2].copy()
        tails = {lag: LowerTail().fit(np.concatenate([motion(global_bank[i], lag) for i in splits['reference']]),
                                     np.concatenate([angles[i] for i in splits['reference']])) for lag in [1, 4]}
        stalls = {i: stall_score(global_bank[i], angles[i], tails) for i in ids}
        full_stalls = {i: hold(np.arange(len(stalls[i])) * 2, stalls[i], lookup[i]['frames']) for i in ids}
        st_limit = float(np.quantile(np.concatenate([full_stalls[i] for i in splits['threshold']]), .995, method='higher'))
        # Verify future invariance at startup, lag, and adaptation boundaries.
        probe = ids[0]
        for n in [1, 4, 5, 23, 24, 25]:
            if n <= len(stalls[probe]):
                np.testing.assert_array_equal(stall_score(global_bank[probe][:n], angles[probe][:n], tails), stalls[probe][:n])
        period, deadline = meta.get('T', 0.), meta.get('D', 0)
        records, per_video, fitted_bank = [], [], {}
        for base_arm, base_limit in limits.items():
            fused = {i: np.maximum(bank[i][base_arm] / max(base_limit, 1e-6), 1.1 * full_stalls[i] / max(st_limit, 1e-6)) for i in ids}
            fitted = dict(base_threshold=float(base_limit), stall_threshold=st_limit,
                          fusion_threshold=operating_threshold(fused, splits, experiment, period, deadline),
                          base_low=float(np.median([low(bank[i][base_arm][::2]) for i in splits['reference']])),
                          fusion_low=float(np.median([low(fused[i][::2]) for i in splits['reference']])))
            fitted_bank[base_arm] = fitted
            for partition, selected in groups.items():
                yy, ss, aa = [], [], []
                for i in selected:
                    base, st = bank[i][base_arm][::2], stalls[i]
                    score, alarm, threshold = predict(base, st, fitted)
                    # The first three variants preserve the source ranking exactly.
                    np.testing.assert_array_equal(score[:, :3], np.repeat(base[:, None], 3, axis=1))
                    for n in [1, 4, 5, 23, 24, 25]:
                        if n <= len(base):
                            for prefix, full in zip(predict(base[:n], st[:n], fitted), (score, alarm, threshold)):
                                np.testing.assert_array_equal(prefix, full[:n])
                    indices = np.arange(len(base)) * 2
                    score, alarm, threshold = [hold(indices, v, lookup[i]['frames']) for v in [score, alarm, threshold]]
                    np.testing.assert_array_equal(score[:, 0], bank[i][base_arm])
                    np.testing.assert_array_equal(alarm[:, 0], bank[i][base_arm] > base_limit)
                    # Labels are opened only after every prediction for this source is fixed.
                    y = labels_for(lookup[i])
                    yy.append(y); ss.append(score); aa.append(alarm)
                    for j, arm in enumerate(ARMS):
                        per_video.append(dict(id=i, partition=partition, base_arm=base_arm, configuration=arm,
                                              **measure(y, score[:, j], alarm[:, j]),
                                              **event_metrics(y, alarm[:, j].astype(float), .5)))
                    path = dest / 'predictions' / base_arm / (i + '.npz')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, scores=score, alarms=alarm, thresholds=threshold, labels=y)
                y, s, a = np.concatenate(yy), np.concatenate(ss), np.concatenate(aa)
                for j, arm in enumerate(ARMS):
                    ev = [r for r in per_video if r['base_arm'] == base_arm and r['partition'] == partition and r['configuration'] == arm]
                    events = sum(r['events'] for r in ev); detected = sum(r['detected_events'] for r in ev)
                    records.append(dict(experiment=experiment, scene=scene, seed=seed, fold=fold,
                                        base_arm=base_arm, configuration=arm, partition=partition,
                                        events=events, detected_events=detected,
                                        event_coverage=detected/events if events else None,
                                        **measure(y, s[:, j], a[:, j])))
        original_metrics = read(root / 'results' / experiment / suffix / 'metrics.json')
        checks, published_deltas = [], []
        published_metrics = read(root.parent / 'references/team_cyclevad_20261016/results' / experiment / suffix / 'metrics.json')
        def matches(row, base_arm, partition):
            if row['partition'] != partition:
                return False
            if experiment == 'E8B':
                return base_arm == row['arm'] + '__' + row['readout']
            if experiment == 'E12':
                return base_arm == f"{row['arm']}_f{int(100*row['fraction'])}__{row['readout']}"
            return base_arm == row['arm']
        for row in [r for r in records if r['configuration'] == 'baseline']:
            local = next(r for r in original_metrics if matches(r, row['base_arm'], row['partition']))
            for key in ['auroc', 'ap', 'fpr', 'recall']:
                if row[key] is None:
                    assert local[key] is None
                else:
                    np.testing.assert_allclose(row[key], local[key], atol=1e-12, rtol=0)
            checks.append(dict(base_arm=row['base_arm'], partition=row['partition'], original_local_metrics_equal=True))
            if row['partition'] == 'historical_test':
                published = next(r for r in published_metrics if matches(r, row['base_arm'], row['partition']))
                published_deltas.append(dict(base_arm=row['base_arm'], **{k: dict(local=row[k], upstream=published[k],
                                         delta_pp=100*(row[k]-published[k]) if row[k] is not None and published[k] is not None else None)
                                         for k in ['auroc','ap','fpr','recall']}))
        dest.mkdir(parents=True, exist_ok=True)
        joblib.dump(dict(tails=tails, fitted=fitted_bank, parameters=PARAMETERS), dest / 'addon.joblib', compress=3)
        write_json(dest / 'metrics.json', records)
        write_json(dest / 'per_video.json', per_video)
        write_json(dest / 'baseline_checks.json', checks)
        write_json(dest / 'upstream_baseline_deltas.json', published_deltas)
        write_json(dest / 'complete.json', dict(status='complete', experiment=experiment, seed=seed,
                   scene=scene, fold=fold, splits=splits, base_arms=list(limits), configurations=ARMS,
                   baseline_scores_and_alarms_exact=True, prefix_checks_passed=True,
                   stall_phase='shared unmodified base tracker angle; not E8B intervention coordinates',
                   parameters=PARAMETERS, inference_labels_used=False))
        print(f'ABLATION {experiment} seed{seed} {scene}/fold{fold}: {len(limits)} x 6 complete', flush=True)

def report(root):
    import csv
    from collections import defaultdict
    rows = []
    for path in sorted((root / 'ablation').glob('*/seed*/R*/fold*/metrics.json')):
        rows += read(path)
    dest = root / 'report'; dest.mkdir(exist_ok=True)
    if not rows:
        return
    with (dest / 'all_metrics.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    keys = ['auroc', 'ap', 'fpr', 'recall', 'precision', 'f1', 'event_coverage']
    groups = defaultdict(list)
    for r in rows:
        if r['partition'] == 'historical_test':
            groups[(r['experiment'], r['seed'], r['base_arm'], r['configuration'])].append(r)
    macro = []
    for (exp, seed, base, arm), values in sorted(groups.items()):
        macro.append(dict(experiment=exp, seed=seed, base_arm=base, configuration=arm,
                          scenes=sorted(r['scene'] for r in values), complete_macro=len(values) == 4,
                          **{k: float(np.mean([r[k] for r in values if r[k] is not None])) if any(r[k] is not None for r in values) else None for k in keys}))
    write_json(dest / 'macro.json', macro)
    lines = ['# E8B / E12 / E11B — six fixed ablations', '',
             'Percent values. Each scene is equally weighted. Partial scene sets are explicitly marked.',
             'Primary baselines were fixed before this run: E8B M3 conditional_only; E12 S1 100% head_only; E11B CP.',
             'All original controls/readouts remain in all_metrics.csv. Historical test: fold 0 only.',
             'Ranking metrics use continuous scores; FPR/Recall/F1 use the actual six alarm policies.', '']
    for exp, primary in PRIMARY.items():
        lines += [f'## {exp} — {primary}', '', '| Configuration | AUROC | AP | FPR | Recall | F1 | Coverage |',
                  '|---|---:|---:|---:|---:|---:|---|']
        for arm, title in zip(ARMS, TITLES):
            matches = [r for r in macro if r['experiment'] == exp and r['seed'] == 42 and r['base_arm'] == primary and r['configuration'] == arm]
            if not matches:
                lines.append(f'| {title} | pending | pending | pending | pending | pending | 0/4 scenes |'); continue
            r = matches[0]
            vals = ['—' if r[k] is None else f'{100*r[k]:.2f}' for k in ['auroc', 'ap', 'fpr', 'recall', 'f1']]
            lines.append('| ' + ' | '.join([title] + vals + [f"{len(r['scenes'])}/4 scenes"]) + ' |')
        lines.append('')
    (dest / 'TABLES.md').write_text('\n'.join(lines), encoding='utf-8')
