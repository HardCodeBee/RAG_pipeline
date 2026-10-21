"""Verify selected historical bytes and archive members. No inference/scoring."""
from pathlib import Path, PurePosixPath
import gzip
import hashlib
import json
import tarfile

root = Path(__file__).resolve().parents[2]
here = Path(__file__).parent
manifest = json.loads((here / 'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
def sha(data): return hashlib.sha256(data).hexdigest()
def safe(name):
    p = PurePosixPath(name)
    assert not p.is_absolute() and '..' not in p.parts and ':' not in name
    return root / Path(*p.parts)
for e in manifest['source_files']:
    data = safe(e['bundle']).read_bytes()
    if 'compressed_sha256' in e:
        assert sha(data) == e['compressed_sha256'], e['bundle']
        data = gzip.decompress(data)
    assert sha(data) == e['sha256'] and len(data) == e['bytes'], e['bundle']
members = 0
for e in manifest['archives']:
    p = safe(e['bundle'])
    assert sha(p.read_bytes()) == e['sha256'], e['bundle']
    expected = {v['source']: v for v in e['members']}
    with tarfile.open(p, 'r:gz') as tf:
        seen = set()
        for member in tf:
            safe(member.name)
            assert member.isfile() and member.name not in seen
            seen.add(member.name)
            d = tf.extractfile(member).read()
            v = expected[member.name]
            assert sha(d) == v['sha256'] and len(d) == v['bytes'], member.name
            members += 1
        assert seen == set(expected)
delivery = here / 'DELIVERY_FILES.json'
if delivery.exists():
    for e in json.loads(delivery.read_text(encoding='utf-8'))['files']:
        d = safe(e['path']).read_bytes()
        assert sha(d) == e['sha256'] and len(d) == e['bytes'], e['path']
print(json.dumps({'passed': True, 'source_files':len(manifest['source_files']),
                  'archives':len(manifest['archives']), 'archive_members':members,
                  'new_API_requests':0,'new_GPU_jobs':0,'new_inference':False}))
