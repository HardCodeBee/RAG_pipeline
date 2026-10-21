"""Infrastructure-only cache repair; immutable v001 methods/results stay intact.

The first pilot remains a genuine v001 run. Existing selected receipts keep
priority; missing exact payloads are supplemented from frozen B0 provenance.
extension240 explicitly excludes the pilot, so no r000 case can be reopened.
"""
import argparse
import hashlib
import json
from pathlib import Path

from . import run as v1
from .cache_catalog_v002 import augment


def prepare_revision():
    plan = v1.prepare()
    here = Path(__file__).resolve().parent
    old_source = here.parent / 'b0_six_single_changes_v1_20261020/B0_SOURCES.frozen.jsonl'
    protocol = {'revision': 'cache_v002', 'method_changes': 0, 'generation_payload_changes': 0,
                'frozen_v001_protocol_sha256': v1.sha(here / 'PROTOCOL.frozen.json'),
                'source_pins': {str(p.relative_to(v1.ROOT)): v1.sha(p) for p in [
                    Path(__file__), here / 'cache_catalog_v002.py', old_source]},
                'cache_priority': 'Preserve every existing selected entry; fill only absent full payload hashes from first-success frozen B0 call order.',
                'old_completed_cases_replaced': False,
                'reason': 'v001 did not index ordinary/rewrite receipts absent from the complete B0 case JSON; the authoritative receipts live in B0_SOURCES.',
                'cost_policy': 'All v001 submissions and unknown reserves remain charged; no refund or relabeling as cache.',
                'global_cap_cny': 300, 'local_subcap_cny': 25}
    v1.freeze(here / 'INFRA_CACHE_V002.frozen.json', protocol)
    pilot = {c['id'] for c in v1.rows(here / 'pilot60.runtime.frozen.jsonl')}
    extension = [c for c in v1.rows(here / 'runtime300.frozen.jsonl') if c['id'] not in pilot]
    assert len(extension) == 240
    target = here / 'extension240.runtime.frozen.jsonl'
    text = ''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in extension)
    if target.exists():
        assert target.read_text(encoding='utf8') == text
    else:
        target.write_text(text, encoding='utf8', newline='\n')
    return plan, protocol


def run(*, sample='extension240', arms=None, round_id=0, dataset=None):
    plan, revision = prepare_revision()
    for path, expected in revision['source_pins'].items():
        assert v1.sha(v1.ROOT / path) == expected, path
    old_catalog, old_rows = v1.source_catalog, v1.rows

    def catalog(ds):
        values, stats = augment(old_catalog(ds), ds)
        v1.log('cache_v002_restore', **stats)
        return values

    def selected_rows(path):
        if sample == 'extension240' and Path(path).resolve() == (v1.HERE / 'runtime300.frozen.jsonl').resolve():
            return old_rows(v1.HERE / 'extension240.runtime.frozen.jsonl')
        return old_rows(path)

    v1.source_catalog, v1.rows = catalog, selected_rows
    try:
        v1.log('cache_v002_dispatch', sample=sample, round=round_id,
               input_mask_sha256=v1.sha(v1.HERE / ('extension240.runtime.frozen.jsonl' if sample == 'extension240' else
                                                 'pilot60.runtime.frozen.jsonl' if sample == 'pilot60' else 'runtime300.frozen.jsonl')),
               algorithm_source_sha256=v1.sha(v1.HERE / 'pipeline.py'))
        return v1.run(plan, sample=sample, arms=arms, round_id=round_id, dataset=dataset)
    finally:
        v1.source_catalog, v1.rows = old_catalog, old_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--sample', choices=['pilot60', 'extension240', '300'], default='extension240')
    parser.add_argument('--arms', nargs='+', choices=v1.ARMS, default=v1.ARMS)
    parser.add_argument('--round', type=int, default=0)
    parser.add_argument('--dataset', choices=['hotpotqa', '2wiki', 'musique'])
    args = parser.parse_args()
    value = {'prepared': True, 'method_changes': 0} if args.prepare and prepare_revision() else run(
        sample=args.sample, arms=args.arms, round_id=args.round, dataset=args.dataset)
    print(json.dumps(value, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
