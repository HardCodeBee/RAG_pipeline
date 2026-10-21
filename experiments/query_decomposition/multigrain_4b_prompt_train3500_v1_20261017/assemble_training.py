"""Export only explicitly reviewed labels. No generation API or training occurs here."""
from collections import Counter
import json,re,statistics
from training_sources import HERE,ROOT,OUT,OLD,rows,norm,digest,sha,write,write_rows
from audit_training_progress import materialize

def collect():
    old_report=materialize();records=rows(OUT/'OLD_REVIEWED.progress.jsonl')
    frozen={r['index']:r for r in rows(OUT/'LABELING_INPUTS.frozen.jsonl')}
    drafts={r['index']:r for r in rows(OUT/'NEW_2WIKI_DRAFTS.unreviewed.jsonl')}
    seen={r['index'] for r in records}
    for pattern in ['hard_*.json','new2wiki_*.json','musique_*.json']:
        for p in sorted((HERE/'training_reviews').glob(pattern)):
            doc=json.loads(p.read_text(encoding='utf-8'));lo,hi=doc['reviewed_range']
            assert not set(range(lo,hi+1))&seen,p.name
            for i in range(lo,hi+1):
                src=frozen[i];key=str(i);reason=doc['quarantine'].get(key)
                if 'targets' in doc:
                    assert key in doc['targets'] or reason,(p.name,i)
                    target=doc['targets'].get(key,[]);draft=None;changed='query_only_authored'
                else:
                    assert i in drafts,(p.name,i)
                    draft=drafts[i]['draft'];target=doc['revisions'].get(key,draft)
                    assert target is not None or reason,(p.name,i)
                    changed='authored_revision' if key in doc['revisions'] else 'reviewed_template_retained'
                records.append({**src,'draft_target':draft,'target':target,'status':'quarantined' if reason else 'accepted','review_method':doc['reviewer'],'review_batch':str(p.relative_to(HERE)),'review_batch_sha256':sha(p),'change_kind':changed,'reason':reason or doc['note']})
                seen.add(i)
    # A later correction never erases the authored/retained target it superseded.
    fixes=HERE/'training_reviews/SECOND_REVIEW_CORRECTIONS.json'
    if fixes.exists():
        corrections=json.loads(fixes.read_text(encoding='utf-8'))
        by_index={r['index']:r for r in records}
        for key,fix in corrections['revisions'].items():
            r=by_index[int(key)];r['first_review_target']=r['target'];r['target']=fix['target'];r['second_review_reason']=fix['reason'];r['second_review_file']=str(fixes.relative_to(HERE));r['second_review_sha256']=sha(fixes)
    return sorted(records,key=lambda r:r['index']),old_report

def main():
    records,old_report=collect();accepted=[r for r in records if r['status']=='accepted'];quarantine=[r for r in records if r['status']=='quarantined']
    errors=[];seen_q={};seen_ids={}
    protect={norm(r['query']) for r in rows(HERE/'EXCLUDE_FROM_TRAINING.jsonl')}
    for r in accepted:
        i=r['index'];q=r['target'];key=norm(r['query'])
        if key in protect:errors.append([i,'prompt100_overlap'])
        if r.get('protected_text_overlap'):errors.append([i,'protected_text_overlap'])
        if r.get('protected_project_split_conflict'):errors.append([i,'hard_explicit_split_conflict'])
        if key in seen_q:errors.append([i,'duplicate_query',seen_q[key]])
        seen_q[key]=i
        if r['id'] in seen_ids:errors.append([i,'duplicate_master_id',seen_ids[r['id']]])
        seen_ids[r['id']]=i
        if not isinstance(q,list) or any(not isinstance(s,str) or not s.strip() or not s.endswith('?') for s in q):errors.append([i,'invalid_question_array']);continue
        if len({norm(s) for s in q})!=len(q):errors.append([i,'duplicate_questions'])
        for s in q:
            if re.search(r'#\d+|UNRESOLVED|the result of \(',s):errors.append([i,'unresolved_placeholder'])
    report={'status':'reviewed_pending_final_QA','accepted_count':len(accepted),'quarantined_count':len(quarantine),'target_count':3500,'mechanical_errors':errors,'by_dataset':dict(Counter(r['source']['dataset'] for r in accepted)),'by_cohort':dict(Counter(r['cohort'] for r in accepted)),'native_2wiki_types':dict(Counter(r['source'].get('type') for r in accepted if r['source']['dataset']=='2wiki' and r['cohort']=='new_train_pool')),'native_musique_hops':dict(Counter(r['source'].get('type') for r in accepted if r['source']['dataset']=='musique' and r['cohort']=='new_train_pool')),'empty_support_arrays':sum(not r['target'] for r in accepted),'support_count_distribution':dict(Counter(len(r['target']) for r in accepted)),'old_source_unchanged':sha(OLD/'accepted_master.jsonl')=='dd68d939b5c3b4b97536f2a2ebe4bdf508cd617922616b12ea1aed3074195545','all_prompt100_excluded':not any(norm(r['query']) in protect for r in accepted),'selected_prompt':'v12','prompt_sha256':sha(HERE/'FINAL_PROMPT.txt'),'selection_is_fresh_test':False,'review_limit':'Current Codex authored and reviewed; some 2Wiki uses inspected templates and every query/parsed span. No independent human gold, guaranteed90% semantic accuracy, retrieval or downstream QA result.','api_calls':0,'training_started':False}
    write_rows(OUT/'REVIEWED_ALL.jsonl',records);write_rows(OUT/'QUARANTINE.jsonl',quarantine);write_rows(OUT/'MASTER_3500.jsonl',accepted);write(OUT/'DATA_REPORT.json',report)
    assert len(accepted)==3500,(len(accepted),'Need additional explicitly reviewed records, never pad or silently drop.')
    assert not errors,errors
    assert report['old_source_unchanged']
    prompt=(HERE/'FINAL_PROMPT.txt').read_text(encoding='utf-8').strip()
    sft=[{'messages':[{'role':'system','content':prompt},{'role':'user','content':r['query']},{'role':'assistant','content':json.dumps(r['target'],ensure_ascii=False,separators=(',',':'))}]} for r in accepted]
    write_rows(OUT/'train3500.messages.jsonl',sft)
    write_rows(OUT/'EXCLUDE_FUTURE_EVALUATION.jsonl',[{'id':r['id'],'source':r['source'],'query':r['query'],'normalized_query_sha256':digest(norm(r['query']))} for r in accepted])
    write_rows(OUT/'OLD_NEW_DIFF.jsonl',[{'index':r['index'],'id':r['id'],'query':r['query'],'status':r['status'],'old_target':r['old_target'],'new_target':r['target'],'review_batch':r['review_batch'],'reason':r['reason'],'second_review_reason':r.get('second_review_reason')} for r in records if r['cohort']=='old2368'])
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
