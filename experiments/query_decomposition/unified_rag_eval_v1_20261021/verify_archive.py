"""Verify same-volume directory identity and preserved frozen source bytes."""
from datetime import datetime,timezone
from pathlib import Path
from .prepare_plan import ROOT,HERE,read,write,sha
from .prepare_archive import DEST,fingerprint

def main():
    index=read(DEST/'INDEX.json')
    for op in index['operations']:
        source,target=Path(op['source']),Path(op['target'])
        assert source.is_dir() and target.is_dir()
        assert source.resolve()==target.resolve(),source
        actual=fingerprint(target)
        # JSON serializes tuples as arrays; normalize the observation likewise.
        import json
        assert json.loads(json.dumps(actual))==op['before'],op['relative']
    protected=read(HERE/'PROTECTED_FILES_BEFORE.json')
    assert all((ROOT/p).exists() and sha(ROOT/p)==h for p,h in protected.items())
    index.update(status='archived_verified',verified_utc=datetime.now(timezone.utc).isoformat(),frozen_sources_unchanged=len(protected),results_deleted=0)
    write(DEST/'INDEX.json',index)
    inventory=read(HERE/'ARCHIVE_INVENTORY.json')
    for item in inventory['directories']:
        if item['classification']=='unrelated_archive_candidate':
            item.update(classification='unrelated_archived',archive_status='physically_moved_with_compatibility_junction',physical_move=True,archive_root=DEST.relative_to(ROOT).as_posix())
    inventory['strategy']='physical move plus index plus original-path compatibility junctions, explicitly approved by user'
    write(HERE/'ARCHIVE_INVENTORY.json',inventory)
    print('PASS: '+str(len(index['operations']))+' moved directory identities and compatibility paths; '+str(len(protected))+' frozen source hashes unchanged.')

if __name__=='__main__':main()
