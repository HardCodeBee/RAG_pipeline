"""Standard-library verification of the delivered bytes; no inference."""
from pathlib import Path,PurePosixPath
import hashlib,json,zipfile

HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def digest(data):return hashlib.sha256(data).hexdigest()
def verify():
 manifest=read(HERE/'DELIVERY_MANIFEST.json')
 for row in manifest['files']:
  p=HERE/row['path'];data=p.read_bytes()
  assert len(data)==row['bytes'] and digest(data)==row['sha256'],row['path']
 archive=read(HERE/'ARCHIVE_MANIFEST.json');p=HERE/archive['archive']
 assert digest(p.read_bytes())==archive['sha256']
 expected={r['path']:r for r in archive['members']}
 with zipfile.ZipFile(p) as z:
  assert len(z.namelist())==len(set(z.namelist()))==archive['member_count']
  assert set(z.namelist())==set(expected)
  assert z.testzip() is None
  for name in z.namelist():
   rel=PurePosixPath(name);assert not rel.is_absolute() and '..' not in rel.parts and '\\' not in name
   data=z.read(name);row=expected[name]
   assert len(data)==row['bytes'] and digest(data)==row['sha256'],name
 cp=read(HERE/'CONDITIONAL_PROBABILITIES.json')
 assert cp['summary']['2wiki']['complete_eligible_queries']==485
 assert cp['summary']['hotpotqa']['complete_eligible_queries']==238
 assert len(read(HERE/'PENDING_FACTS.json')['items'])==7
 cross=read(HERE/'SOURCE_CROSSWALK.json')
 assert cross['exact_raw_source_matches']==len(cross['records'])==760
 result={'passed':True,'delivery_files_verified':len(manifest['files']),'archive_members_verified':len(expected),'complete_queries':723,'source_crosswalk_matches':760,'pending_goal_facts':7,'new_inference':False}
 return result
if __name__=='__main__':print(json.dumps(verify(),ensure_ascii=False))
