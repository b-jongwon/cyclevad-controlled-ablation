"""Verify published numerical artifacts, source provenance and all 180 units."""
from pathlib import Path
import gzip
import hashlib
import itertools
import json

ROOT = Path(__file__).resolve().parents[1]

def data(path):
    if not path.exists() and path.suffix == '.json':
        path = path.with_suffix('.json.gz')
    raw = path.read_bytes()
    return gzip.decompress(raw) if path.suffix == '.gz' else raw

def read(path):
    return json.loads(data(path))

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def main():
    manifest=read(ROOT/'provenance/artifact_manifest.json')
    for item in manifest:
        path=ROOT/item['published'];raw=path.read_bytes()
        assert len(raw)==item['stored_bytes'] and digest(raw)==item['stored_sha256'], str(path)
        if not item.get('formatting_paths_adjusted'):
            original=gzip.decompress(raw) if item['gzip'] else raw
            assert len(original)==item['original_bytes'] and digest(original)==item['original_sha256'], str(path)
    source=read(ROOT/'provenance/upstream_source_manifest.json')
    for name,expected in source['sha256'].items():
        assert digest((ROOT/name).read_bytes())==expected, name
    protocol=read(ROOT/'provenance/protocol.json')
    for name,expected in protocol['addon_hashes'].items():
        assert digest((ROOT/'provenance/original_execution'/name).read_bytes())==expected, name
    for name,expected in protocol['source_hashes'].items():
        assert digest((ROOT/name.replace('\\','/')).read_bytes())==expected, name
    completion=read(ROOT/'provenance/all_completion.json')
    assert completion['status']=='complete'
    assert completion['seeds']==[42,43,44] and completion['folds']==list(range(5))
    assert completion['scenes']==['R01','R02','R03','R04']
    total,checks=0,0
    for experiment,seed,scene,fold in itertools.product(['E8B','E12','E11B'],[42,43,44],['R01','R02','R03','R04'],range(5)):
        folder=ROOT/'results/ablation'/experiment/f'seed{seed}'/scene/f'fold{fold}'
        c=read(folder/'complete.json');b=read(folder/'baseline_checks.json')
        assert c['status']=='complete' and c['baseline_scores_and_alarms_exact'] and c['prefix_checks_passed'], str(folder)
        assert len(c['configurations'])==6 and all(v['original_local_metrics_equal'] for v in b),str(folder)
        total+=1;checks+=len(b)
    primary={'E8B':'M3__conditional_only','E12':'S1_f100__head_only','E11B':'CP'}
    rows=[r for r in read(ROOT/'reports/macro.json') if r['base_arm']==primary[r['experiment']]]
    assert len(rows)==54 and all(r['complete_macro'] and r['scenes']==completion['scenes'] for r in rows)
    assert total==180 and checks==15808
    print(json.dumps(dict(status='passed',numeric_artifacts=len(manifest),experiment_units=total,baseline_metric_checks=checks,primary_macro_rows=len(rows),upstream_source_hashes='matched',original_execution_hashes='matched')))

if __name__=='__main__':main()
