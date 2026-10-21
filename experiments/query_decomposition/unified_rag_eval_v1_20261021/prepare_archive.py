"""Prepare the explicitly approved physical archive with compatibility junctions."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
from .prepare_plan import ROOT,HERE,read,write,rel

DEST=ROOT/'archives/rag_evaluation_history_20261021'

def fingerprint(p):
    root=p.stat()
    files=[]
    for base,dirs,names in os.walk(p,followlinks=False):
        for name in names:
            f=Path(base)/name;s=f.stat()
            files.append((f.relative_to(p).as_posix(),s.st_size,s.st_mtime_ns,s.st_ino))
    return {'directory_file_id':root.st_ino,'file_count':len(files),'total_bytes':sum(x[1] for x in files),'files':sorted(files)}

def main():
    inventory=read(HERE/'ARCHIVE_INVENTORY.json')
    operations=[]
    for item in inventory['directories']:
        if item['classification']!='unrelated_archive_candidate':continue
        for relative in [item['path'],*item['outputs']]:
            source=ROOT/relative;target=DEST/'payload'/relative
            # Verify the actual absolute source and target before any recursive move.
            assert source.resolve().is_relative_to(ROOT.resolve())
            assert target.resolve().is_relative_to(DEST.resolve())
            assert source.resolve()!=ROOT.resolve() and not source.is_symlink()
            if target.exists():raise RuntimeError('Archive destination already exists: '+str(target))
            operations.append({'source':str(source),'target':str(target),'relative':relative,'before':fingerprint(source)})
    DEST.mkdir(parents=True,exist_ok=True)
    manifest={'utc':datetime.now(timezone.utc).isoformat(),'authority':'User approved physical move plus index and compatibility junctions in this chat','archive_format':'moved_directories_with_original_path_junctions','workspace':str(ROOT),'archive_root':str(DEST),'status':'prepared_not_moved','operations':operations}
    write(DEST/'INDEX.json',manifest)
    print(json.dumps({'operations':len(operations),'experiment_directories':sum(x['relative'].startswith('experiments/') for x in operations),'bytes':sum(x['before']['total_bytes'] for x in operations),'index':str(DEST/'INDEX.json')},indent=2))

if __name__=='__main__':main()
