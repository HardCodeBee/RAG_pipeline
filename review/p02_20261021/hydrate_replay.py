"""Restore two historical replay files only. No model/API execution."""
from pathlib import Path
import gzip
import hashlib
import json

root = Path(__file__).resolve().parents[2]
manifest = json.loads((Path(__file__).parent / 'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
for entry in manifest['source_files']:
    if entry['transformation'] != 'lossless gzip; hydrate to original path':
        continue
    packed = root / entry['bundle']
    assert hashlib.sha256(packed.read_bytes()).hexdigest() == entry['compressed_sha256']
    data = gzip.decompress(packed.read_bytes())
    assert hashlib.sha256(data).hexdigest() == entry['sha256']
    target = root / entry['source']
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != entry['sha256']:
            raise FileExistsError('Refusing to overwrite a different file: ' + str(target))
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    print('Verified historical replay asset:', target.relative_to(root).as_posix())
print('No API, GPU, retrieval or evaluation execution was started.')
