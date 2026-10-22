"""Pure JSONL bridge contract, with no launcher, socket, SSH or GPU import.

An approved future orchestrator may pass streams of a separately allocated
retrieval-only worker. Credentials and LLM messages never cross this bridge.
"""

from copy import deepcopy
import hashlib
import json

PROTOCOL = "p02-dynamic-retrieval-jsonl-draft-v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def identity(resource_identity, dataset, qid, query, k=10):
    if dataset not in ("hotpotqa", "2wiki", "musique") or type(k) is not int or k != 10:
        raise ValueError("Fixed90 dataset/k contract")
    if not all(isinstance(x, str) and x.strip() for x in (resource_identity, qid, query)):
        raise ValueError("Empty retrieval identity")
    return dict(protocol=PROTOCOL, resource_identity=resource_identity,
        dataset=dataset, qid=qid, query=query, k=k)


def key(request):
    return hashlib.sha256(canonical(request).encode("utf-8")).hexdigest()


def check_packet(request, packet):
    if any(packet.get(field) != request[field] for field in ("qid", "query", "k")):
        raise ValueError("Retrieval packet query/qid/k mismatch")
    docs = packet.get("docs")
    if not isinstance(docs, list) or len(docs) != 10 or len({d["doc_id"] for d in docs}) != 10:
        raise ValueError("Need10 unique actual route docs")
    for rank, doc in enumerate(docs, 1):
        if doc.get("rank") != rank or not all(field in doc for field in ("source", "text", "block", "score")):
            raise ValueError("Missing ranked original document fields")
        if doc["block"] != f'[Chunk {rank} | source={doc["source"]}, pages=1-1]\n{doc["text"]}':
            raise ValueError("Changed block formatting")
    return deepcopy(packet)


class PacketRouter:
    """Per-flow exact-query cache. fetch receives only the request dictionary.

    Optional recorded packets must ALREADY have a separately verified matching
    physical resource identity. This class cannot establish cross-path resource
    equivalence and does not read old SQLite DBs or pick a better answer.
    """
    def __init__(self, resource_identity, dataset, qid, fetch, *, saved=(), event=None):
        self.resource_identity, self.dataset, self.qid = resource_identity, dataset, qid
        self.fetch, self.event, self.cache = fetch, event or (lambda event: None), {}
        for packet in saved:
            request = identity(resource_identity, dataset, qid, packet["query"])
            self.cache.setdefault(key(request), check_packet(request, packet))

    def __call__(self, query):
        request = identity(self.resource_identity, self.dataset, self.qid, query)
        request_id = key(request)
        hit = request_id in self.cache
        if not hit:
            self.cache[request_id] = check_packet(request, self.fetch(deepcopy(request)))
        self.event(dict(event="retrieve", request_id=request_id,
            source="exact_resource_query_cache" if hit else "worker_search", **request))
        return deepcopy(self.cache[request_id])


def client_exchange(request, writer, reader):
    """Strict synchronous one-inflight stream exchange; no reconnect/retry here."""
    request_id = key(request)
    writer.write(canonical(dict(request_id=request_id, **request)) + "\n")
    writer.flush()
    line = reader.readline()
    if not line:
        raise RuntimeError("Retrieval channel closed; preserve query, no API-side continuation")
    response = json.loads(line)
    if response.get("request_id") != request_id or response.get("identity") != request:
        raise ValueError("Worker identity mismatch")
    if response.get("status") != "success":
        raise RuntimeError("Worker held/error; no fabricated empty packet")
    return check_packet(request, response["packet"])


def serve_one(line, fetch, *, expected_resources):
    """Injected worker backend must use frozen BGE+corpus; no real loader here."""
    request = json.loads(line)
    request_id = request.pop("request_id")
    if request != identity(request["resource_identity"], request["dataset"], request["qid"], request["query"], request["k"]):
        raise ValueError("Unexpected retrieval fields or protocol")
    if request_id != key(request) or expected_resources.get(request["dataset"]) != request["resource_identity"]:
        raise ValueError("Unapproved resource identity")
    packet = check_packet(request, fetch(deepcopy(request)))
    return canonical(dict(status="success", request_id=request_id, identity=request, packet=packet)) + "\n"
