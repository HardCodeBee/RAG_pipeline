"""All-split global paragraph union; resumable dense/BM25 immutable resources."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from .common import ROOT, HERE, RUN, DATA, read, rows, save, sha, log
from .prepare_samples import h

def iter_array(path):
    decoder=json.JSONDecoder();buffer='';started=False;ended=False
    with Path(path).open(encoding='utf8') as f:
        while not ended:
            part=f.read(2*1024*1024);buffer+=part
            if not started:
                buffer=buffer.lstrip()
                if not buffer.startswith('['):raise ValueError('Expected native JSON array')
                buffer=buffer[1:];started=True
            while True:
                buffer=buffer.lstrip()
                if buffer.startswith(','):buffer=buffer[1:].lstrip()
                if buffer.startswith(']'):ended=True;break
                if not buffer:break
                try:value,stop=decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    if not part:raise
                    break
                yield value
                buffer=buffer[stop:]
            if not part and not ended:raise ValueError('Truncated native JSON array')

def records(path):
    if path.suffix=='.json':yield from iter_array(path)
    else:
        with path.open(encoding='utf8') as f:
            for line in f:
                if line.strip():yield json.loads(line)

def paragraphs(dataset):
    native=DATA/dataset/'native'
    names=['train.json','dev.json','test.json'] if dataset=='2wiki' else ['musique_'+kind+'_v1.0_'+split+'.jsonl' for kind in ['ans','full'] for split in ['dev','test','train']]
    for name in names:
        path=native/name
        for r in records(path):
            if dataset=='2wiki':
                for title,sents in r['context']:yield title,' '.join(sents)
            else:
                for p in r['paragraphs']:yield p['title'],p['paragraph_text']

def build(dataset):
    corpus=DATA/dataset/'corpus';corpus.mkdir(parents=True,exist_ok=True)
    status=RUN/(dataset+'_index_progress.json')
    if not (corpus/'CORPUS_FROZEN.json').exists():
        save(status,{'status':'global_corpus_union','dataset':dataset})
        seen=set();offsets=[]
        partial=corpus/'chunks.jsonl.partial'
        with partial.open('wb') as f:
            for title,text in paragraphs(dataset):
                if not title.strip() or not text.strip():raise ValueError('Empty native paragraph')
                doc=h(title+'\n'+text)
                if doc in seen:continue
                seen.add(doc);vid=len(offsets);offsets.append(f.tell())
                record={'chunk_id':doc,'vector_id':vid,'doc_id':doc,'source':title,'page_start':1,'page_end':1,
                        'text':title+'\n'+text,'token_count':max(1,len((title+' '+text).split()))}
                f.write((json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf8'))
                if vid%20000==0:save(status,{'status':'global_corpus_union','dataset':dataset,'unique_documents':vid+1})
        partial.replace(corpus/'chunks.jsonl');np.save(corpus/'chunk_offsets.npy',np.asarray(offsets,dtype=np.int64))
        chunks=corpus/'chunks.jsonl'
        manifest={'build_id':'comprehensive_'+dataset+'_global_v1','status':'complete','dataset':dataset,
          'documents':len(offsets),'deduplication':'sha256(title + newline + paragraph_text); preserve raw paragraph',
          'scope':'all official associated train/dev/test contexts; MuSiQue includes Ans and Full paragraphs',
          'question_conditioned_candidate_corpus':False,
          'artifacts':{'chunks':{'file':'chunks.jsonl','rows':len(offsets),'sha256':sha(chunks),'size_bytes':chunks.stat().st_size}},
          'offsets_sha256':sha(corpus/'chunk_offsets.npy'),
          'sources':{p.name:sha(p) for p in sorted((DATA/dataset/'native').iterdir()) if p.name!='id_aliases.json'}}
        save(corpus/'CORPUS_FROZEN.json',manifest)
        log('global_corpus_frozen',dataset=dataset,documents=len(offsets),manifest_sha256=sha(corpus/'CORPUS_FROZEN.json'))
    manifest=read(corpus/'CORPUS_FROZEN.json');n=manifest['documents']
    if sha(corpus/'chunks.jsonl')!=manifest['artifacts']['chunks']['sha256']:raise ValueError('Corpus changed')
    base=read(ROOT/'experiments/query_decomposition/query_form_ablation_v1/resources.json');spec=base['dense']['bge']
    dense=corpus/'dense';dense.mkdir(exist_ok=True)
    if not (dense/'DENSE_FROZEN.json').exists():
        import torch
        from src.embedders.text_embedder import TextEmbedder
        torch.set_num_threads(4)
        encoder=TextEmbedder(backend='sentence_transformers',model_name=spec['query_model'],revision=spec['query_revision'],
          normalize=True,batch_size=64,query_prefix='',document_prefix='',max_sequence_length=512,local_files_only=True,device='cuda')
        if encoder.embedding_space('inner_product').to_dict()!=spec['space']:raise ValueError('Shared BGE space changed')
        started=time.perf_counter();batch=[];part_idx=0;total=0;files={}
        def part(texts):
            nonlocal part_idx,total
            path=dense/f'part-{part_idx:05d}.npy';expected=(len(texts),384)
            if path.exists():
                old=np.load(path,mmap_mode='r');assert old.shape==expected and old.dtype==np.float32
                del old
            else:
                vectors=encoder.encode_documents(texts)
                tmp=path.with_suffix('.partial.npy');np.save(tmp,vectors);tmp.replace(path)
            files[path.name]=sha(path);total+=len(texts);part_idx+=1
            save(status,{'status':'dense_encoding','dataset':dataset,'encoded_documents':total,'planned':n,
                         'seconds':time.perf_counter()-started})
            print(dataset,'encoded',total,'/',n,flush=True)
        with (corpus/'chunks.jsonl').open(encoding='utf8') as f:
            for line in f:
                batch.append(json.loads(line)['text'])
                if len(batch)==10000:part(batch);batch=[]
        if batch:part(batch)
        if total!=n:raise ValueError('Incomplete dense encoding')
        save(dense/'DENSE_FROZEN.json',{'documents':n,'dimension':384,'space':spec['space'],'part_sha256':files,
            'source_sha256':manifest['artifacts']['chunks']['sha256'],'embedding_batch_size':64,'part_size':10000})
    sparse=corpus/'bm25'
    save(status,{'status':'building_bm25','dataset':dataset,'documents':n})
    from src.persistence.artifact_validation import VerifiedBuild
    from src.retrievers.sqlite_bm25 import build_sqlite_bm25_index
    verified=VerifiedBuild(corpus,manifest,{'chunks':corpus/'chunks.jsonl'})
    sparse_artifact=build_sqlite_bm25_index(verified,sparse,k1=1.5,b=.75,transaction_documents=1000)
    sparse_manifest=read(sparse/'manifest.json')
    resources={'corpus':{'dataset':dataset,'documents':n,'manifest':str(corpus/'CORPUS_FROZEN.json'),
       'manifest_sha256':sha(corpus/'CORPUS_FROZEN.json'),'chunks':str(corpus/'chunks.jsonl'),
       'chunks_sha256':manifest['artifacts']['chunks']['sha256'],'offsets':str(corpus/'chunk_offsets.npy')},
      'dense':{'bge':{**spec,'build':str(dense),'manifest_name':'DENSE_FROZEN.json','manifest_sha256':sha(dense/'DENSE_FROZEN.json')}},
      'bm25':{'directory':str(sparse),'manifest_sha256':sha(sparse/'manifest.json'),
        'sparse_index_id':sparse_manifest['sparse_index_id'],'backend':'sqlite_bm25_v1','analyzer':'english_regex_casefold_v1','k1':1.5,'b':.75}}
    save(HERE/'resources_frozen'/f'{dataset}.json',resources)
    save(status,{'status':'complete','dataset':dataset,'documents':n,'resources':str(HERE/'resources_frozen'/f'{dataset}.json')})
    log('dataset_indexes_ready',dataset=dataset,documents=n,resources_sha256=sha(HERE/'resources_frozen'/f'{dataset}.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['2wiki','musique']);a=p.parse_args();build(a.dataset)
