"""Safely extract review sources to a new empty directory; runs no research code."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',required=True)
    args=ap.parse_args()
    package=Path(__file__).resolve().parent
    target=Path(args.output).resolve()
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError('Choose a new or empty output directory; existing files are never overwritten.')
    manifest=json.loads((package/'ARCHIVE_MANIFEST.json').read_text(encoding='utf-8'))
    target.mkdir(parents=True,exist_ok=True)
    count=0
    for record in manifest['archives']:
        archive=package/record['path']
        if hashlib.sha256(archive.read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError('Archive hash mismatch')
        with zipfile.ZipFile(archive) as z:
            for row in record['members']:
                name=PurePosixPath(row['source'])
                if name.is_absolute() or '..' in name.parts or ':' in str(name):
                    raise ValueError('Unsafe member path')
                p=target.joinpath(*name.parts).resolve()
                if not p.is_relative_to(target) or p.exists():
                    raise ValueError('Unsafe or duplicate destination')
                b=z.read(row['source'])
                if len(b)!=row['bytes'] or hashlib.sha256(b).hexdigest()!=row['sha256']:
                    raise ValueError('Member hash mismatch')
                p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);count+=1
    print(json.dumps({'extracted_files':count,'destination':str(target),'research_executed':False}))

if __name__=='__main__': main()
