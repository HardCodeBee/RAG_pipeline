"""Only M0 and M4; no gold, score, query lexical ranking or external calls."""
from copy import deepcopy
import re
from p02_multigrain_timing_cachefix_v1.core.pipeline import merge

ARMS=("M0","M4")


def fingerprint(doc):
    return doc["source"],tuple(re.findall(r"\w+|[^\w\s]",doc["text"]))


def union(original,rewritten):
    result=[];seen=set()
    for rank in range(10):
        for route in(original,rewritten):
            if route and rank<len(route["docs"]):
                doc=route["docs"][rank]
                if doc["doc_id"]not in seen:result.append(doc);seen.add(doc["doc_id"])
    assert len(result)>=10
    return result


def packet(original,rewritten,arm):
    if arm not in ARMS:raise ValueError(arm)
    old=merge(original,rewritten)
    if arm=="M0"or rewritten is None:return old
    pool=union(original,rewritten)
    if len(pool)==10:return old
    chosen=[];seen=set()
    for doc in pool:
        key=fingerprint(doc)
        if key not in seen:
            chosen.append(doc);seen.add(key)
            if len(chosen)==10:break
    ids={doc["doc_id"]for doc in chosen}
    chosen +=[doc for doc in pool if doc["doc_id"]not in ids][:10-len(chosen)]
    position={doc["doc_id"]:i for i,doc in enumerate(pool)}
    chosen.sort(key=lambda doc:position[doc["doc_id"]])
    assert len(chosen)==len({doc["doc_id"]for doc in chosen})==10
    docs=[{**deepcopy(doc),"rank":i,"block":f'[Chunk {i} | source={doc["source"]}, pages=1-1]\n{doc["text"]}'}for i,doc in enumerate(chosen,1)]
    return {**old,"docs":docs,"merge_policy":"same_source_format_duplicate_fill_from_interleave_top10"}
