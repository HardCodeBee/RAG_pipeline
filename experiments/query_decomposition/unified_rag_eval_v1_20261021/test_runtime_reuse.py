"""Synthetic sqlite tests only: no provider, model, index, or old-state writes."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from experiments.query_decomposition.unified_rag_eval_v1_20261021.runtime_reuse import ReadOnlyReuseCatalog, discover_sources


METHOD = 'b00_rag_top10_v1'
RESOURCES = {'corpus': {'dataset': 'hotpotqa', 'documents': 10, 'manifest_sha256': 'corpus', 'chunks_sha256': 'chunks'},
             'dense': {'bge': {'manifest_sha256': 'index', 'query_model': 'bge', 'query_revision': 'fixed', 'space': {'dimension': 384}}}}
PAYLOAD = {'model': 'fixed-reader', 'messages': [{'role': 'user', 'content': 'Read complete raw evidence. Question?'}],
           'temperature': 0, 'extra_body': {'enable_thinking': False}, 'max_tokens': 32768}
RESULT = {'status': 'success', 'answer': 'answer', 'finish_reason': 'stop', 'returned_model': 'fixed-reader',
          'usage': {'prompt_tokens': 12, 'completion_tokens': 2}}
PACKET = {'qid': 'q', 'query': 'Question?', 'k': 1, 'query_tokens': 5, 'encoder_truncated': False,
          'docs': [{'rank': 1, 'doc_id': 'd', 'source': 'title', 'text': 'raw evidence', 'block': '[Chunk 1]\nraw evidence'}]}


def database(path, identity=None, *, request=True, result=None, status='success', payload=None, qid='q', stage='final', packets=True):
    db = sqlite3.connect(path)
    db.executescript('CREATE TABLE data(kind TEXT,key TEXT,value TEXT,PRIMARY KEY(kind,key));'
                     'CREATE TABLE requests(id TEXT PRIMARY KEY,stage TEXT,payload TEXT,status TEXT,result TEXT);'
                     'CREATE TABLE attempts(id INTEGER PRIMARY KEY,request_id TEXT,status TEXT,result TEXT);')
    if identity is None:
        identity = {'method': METHOD, 'dataset': 'hotpotqa', 'resources': RESOURCES}
    db.execute('INSERT INTO data VALUES(?,?,?)', ('meta', 'identity', json.dumps(identity)))
    if request:
        old = deepcopy(payload or PAYLOAD)
        old['max_tokens'] = 64
        db.execute('INSERT INTO requests VALUES(?,?,?,?,?)', ('receipt', stage, json.dumps(old), status, json.dumps(result if result is not None else RESULT)))
        db.execute('INSERT INTO attempts VALUES(?,?,?,?)', (1, 'receipt', status, json.dumps(result if result is not None else RESULT)))
        db.execute('INSERT INTO data VALUES(?,?,?)', ('token_call', 'binding', json.dumps({'qid': qid, 'stage': stage, 'request_id': 'receipt'})))
    if packets:
        db.execute('INSERT INTO data VALUES(?,?,?)', ('retrieval', 'packet', json.dumps(PACKET)))
    db.commit()
    return db


@pytest.fixture
def pair(tmp_path):
    old = tmp_path / 'old.sqlite3'
    database(old).close()
    current = database(tmp_path / 'current.sqlite3', request=False)
    yield old, SimpleNamespace(db=current)
    current.close()


def lookup(old, current, **kwargs):
    cat = ReadOnlyReuseCatalog([old], method=METHOD, dataset='hotpotqa', resources=RESOURCES)
    return cat.lookup(kwargs.get('qid', 'q'), kwargs.get('stage', 'final'), kwargs.get('payload', PAYLOAD), current)


def test_c1_same_scope_reuses_complete_receipt_without_changing_old_bytes(pair):
    old, current = pair
    before = old.read_bytes()
    value = lookup(old, current)
    assert not value['blockers']
    assert value['candidate']['result'] == RESULT
    assert value['candidate']['payload']['max_tokens'] == 64
    assert value['candidate']['c1_validation']['current_retrieval_packets_verified'] == 1
    assert old.read_bytes() == before


@pytest.mark.parametrize('change', ['question', 'settings', 'qid', 'stage', 'retrieval', 'resources', 'missing_usage', 'length', 'binding_cap', 'returned_model'])
def test_c1_rejects_changed_or_incomplete_evidence(pair, change):
    old, current = pair
    opts = {}
    if change in {'question', 'settings'}:
        payload = deepcopy(PAYLOAD)
        if change == 'question': payload['messages'][0]['content'] += ' Changed'
        else: payload['temperature'] = 1
        opts['payload'] = payload
    elif change in {'qid', 'stage'}:
        opts[change] = 'different'
    elif change == 'retrieval':
        p = deepcopy(PACKET); p['docs'][0]['text'] = 'changed'
        current.db.execute("UPDATE data SET value=? WHERE kind='retrieval'", (json.dumps(p),)); current.db.commit()
    else:
        with sqlite3.connect(old) as db:
            if change == 'resources':
                value = json.loads(db.execute("SELECT value FROM data WHERE kind='meta'").fetchone()[0])
                value['resources']['dense']['bge']['query_revision'] = 'changed'
                db.execute("UPDATE data SET value=? WHERE kind='meta'", (json.dumps(value),))
            else:
                result = deepcopy(RESULT)
                if change == 'missing_usage': del result['usage']
                elif change == 'length': result['finish_reason'] = 'length'
                elif change == 'binding_cap': result['usage']['completion_tokens'] = 64
                elif change == 'returned_model': result['returned_model'] = 'other'
                db.execute('UPDATE requests SET result=?', (json.dumps(result),))
    assert lookup(old, current, **opts)['candidate'] is None


@pytest.mark.parametrize('status', ['unknown', 'in_flight'])
def test_historical_unknown_blocks_only_same_logical_scope_even_if_payload_changes(pair, status):
    old, current = pair
    with sqlite3.connect(old) as db:
        db.execute('UPDATE requests SET status=?,result=NULL', (status,))
    value = lookup(old, current)
    assert value['candidate'] is None and len(value['blockers']) == 1
    assert value['blockers'][0]['reserved_unknown_cny'] is None
    changed = deepcopy(PAYLOAD); changed['messages'][0]['content'] += ' changed upstream'
    assert len(lookup(old, current, payload=changed)['blockers']) == 1
    assert not lookup(old, current, qid='other')['blockers']
    assert not lookup(old, current, stage='other')['blockers']


def test_old_unknown_attempt_is_not_cleared_by_later_request_success(pair):
    old, current = pair
    with sqlite3.connect(old) as db:
        db.execute("INSERT INTO attempts VALUES(2,'receipt','unknown',NULL)")
    value = lookup(old, current)
    assert value['candidate'] is None and value['blockers']


def test_claimed_validation_does_not_replace_db_identity(pair):
    old, current = pair
    with sqlite3.connect(old) as db:
        db.execute("UPDATE data SET value='{}' WHERE kind='meta'")
    cat = ReadOnlyReuseCatalog([{'path': old, 'method': METHOD, 'dataset': 'hotpotqa', 'validated': True}],
                              method=METHOD, dataset='hotpotqa', resources=RESOURCES)
    with pytest.raises(ValueError, match='identity is unverified'):
        cat.lookup('q', 'final', PAYLOAD, current)


def test_trace_binds_precompletion_unknown_when_token_record_is_missing(pair):
    old, current = pair
    with sqlite3.connect(old) as db:
        db.execute("DELETE FROM data WHERE kind='token_call'")
        db.execute("UPDATE requests SET status='unknown',result=NULL")
    old.with_name('trace.jsonl').write_text(json.dumps({'event': 'logical_api_call', 'qid': 'q', 'stage': 'final', 'request_id': 'receipt'}) + '\n', encoding='utf8')
    assert lookup(old, current)['blockers']


def test_separate_question_binding_keeps_shared_request_ownership(pair):
    old, current = pair
    with sqlite3.connect(old) as db:
        db.execute("UPDATE requests SET status='unknown',result=NULL")
        db.execute('INSERT INTO data VALUES(?,?,?)', ('question_request_origin', 'second', json.dumps({'qid': 'q2', 'request_id': 'receipt'})))
    assert lookup(old, current, qid='q2')['blockers']
    assert not lookup(old, current, qid='q3')['blockers']


def test_frozen_resource_file_hash_is_checked_and_p02_protocol_pins_scope(tmp_path):
    protocol_dir = tmp_path / 'p02_formal1500_v1_20261021'; protocol_dir.mkdir()
    protocol = protocol_dir / 'PROTOCOL.frozen.json'
    method = 'p02_multigrain_timing_cachefix_v1'
    protocol.write_text(json.dumps({'method': method, 'resources': {'hotpotqa': RESOURCES}}), encoding='utf8')
    identity = {'task': protocol_dir.name, 'dataset': 'hotpotqa', 'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest()}
    old = tmp_path / 'old.sqlite3'; database(old, identity).close()
    current = database(tmp_path / 'current.sqlite3', request=False)
    try:
        cat = ReadOnlyReuseCatalog([{'path': old, 'protocol_file': protocol}], method=method, dataset='hotpotqa', resources=RESOURCES)
        assert cat.lookup('q', 'final', PAYLOAD, SimpleNamespace(db=current))['candidate']
        protocol.write_text('{}', encoding='utf8')
        with pytest.raises(ValueError, match='identity is unverified'):
            cat.lookup('q', 'final', PAYLOAD, SimpleNamespace(db=current))
    finally:
        current.close()


def test_discovery_does_not_include_other_methods_or_development(tmp_path):
    root = tmp_path / 'outputs/query_decomposition/hotpot_formal_eval_v1/comprehensive_v1_20261015'
    for suffix in [f'main/hotpotqa/formal/{METHOD}/fixed', 'main/hotpotqa/formal/other/fixed', f'main/hotpotqa/dev/{METHOD}/fixed']:
        path = root / 'baseline_integrity_repair_v2_20261017' / suffix / 'checkpoint.sqlite3'
        path.parent.mkdir(parents=True); path.write_bytes(b'')
    found = discover_sources(tmp_path, METHOD, 'hotpotqa')
    assert len(found) == 1 and f'/formal/{METHOD}/' in found[0]['path'].as_posix()


def test_method_bound_external_network_unknown_is_found_without_cross_method_import(pair, tmp_path):
    old, current = pair
    network = tmp_path / 'network.sqlite3'
    database(network, {}, status='unknown').close()
    with sqlite3.connect(old) as db:
        db.execute('DELETE FROM requests')
        call = {'qid': 'q', 'stage': 'final', 'request_id': 'receipt', 'source': str(network), 'status': 'unknown'}
        db.execute('INSERT INTO data VALUES(?,?,?)', ('logical_call', 'external', json.dumps(call)))
    value = lookup(old, current)
    assert value['candidate'] is None and len(value['blockers']) == 1
    assert value['blockers'][0]['binding_db'] == str(old.resolve())
    assert not lookup(old, current, qid='other')['blockers']


def test_historical_actual_attempt_count_and_mirrored_sources_are_deduplicated(pair, tmp_path):
    old, current = pair
    with sqlite3.connect(old) as db:
        db.execute("INSERT INTO attempts VALUES(2,'receipt','failed',NULL)")
        db.execute("INSERT INTO attempts VALUES(3,'receipt','failed',NULL)")
        db.execute("INSERT INTO attempts VALUES(4,'receipt','failed',NULL)")
    mirror = tmp_path / 'mirror.sqlite3'
    db = database(mirror)
    db.execute('DELETE FROM attempts')
    db.execute('INSERT INTO data VALUES(?,?,?)', ('request_origin', 'receipt', json.dumps({'origin_db': str(old), 'origin_request_id': 'receipt'})))
    db.commit(); db.close()
    cat = ReadOnlyReuseCatalog([mirror, old], method=METHOD, dataset='hotpotqa', resources=RESOURCES)
    value = cat.lookup('q', 'final', PAYLOAD, current)
    assert len(value['historical_attempts']) == 4
    assert len({r['identity'] for r in value['historical_attempts']}) == 4
    assert not value['attempt_count_unverified']


def test_missing_historical_attempt_count_blocks_fresh_but_not_complete_c1_reuse(pair):
    old, current = pair
    with sqlite3.connect(old) as db:
        db.execute('DELETE FROM attempts')
    value = lookup(old, current)
    assert value['candidate'] and value['attempt_count_unverified'] and not value['blockers']
    p = deepcopy(PACKET); p['docs'][0]['text'] = 'changed'
    current.db.execute("UPDATE data SET value=? WHERE kind='retrieval'", (json.dumps(p),)); current.db.commit()
    value = lookup(old, current)
    assert value['candidate'] is None and value['blockers'][0]['status'] == 'historical_attempt_count_unverified'


def test_overflow_metadata_requires_actual_full_window_coverage(pair):
    old, current = pair
    packet = deepcopy(PACKET)
    packet.update(query_tokens=1202, content_tokens=1200, processed_content_tokens=1200,
                  encoded_window_tokens=[512, 512, 182], window_count=3)
    with sqlite3.connect(old) as db:
        db.execute("UPDATE data SET value=? WHERE kind='retrieval'", (json.dumps(packet),))
    current.db.execute("UPDATE data SET value=? WHERE kind='retrieval'", (json.dumps(packet),)); current.db.commit()
    assert lookup(old, current)['candidate']
    packet['processed_content_tokens'] = 510
    with sqlite3.connect(old) as db:
        db.execute("UPDATE data SET value=? WHERE kind='retrieval'", (json.dumps(packet),))
    assert lookup(old, current)['candidate'] is None


def recovery_overlay(path, attempt_id):
    from experiments.query_decomposition.unified_rag_eval_v1_20261021.runtime_reuse import _value_sha
    with sqlite3.connect(path) as db:
        payload, status, raw = db.execute("SELECT payload,status,result FROM requests WHERE id='receipt'").fetchone()
        overlay = {'evidence': {'origin_db': str(path.resolve()), 'request_id': 'receipt', 'attempt_id': attempt_id,
            'verified': True, 'outcome': 'verified_not_submitted', 'provider_confirmed_no_charge': True,
            'provider_receipt_id': 'synthetic-provider-evidence', 'source': 'synthetic-proof',
            'payload_sha256': _value_sha(json.loads(payload))}, 'original_request_status': status,
            'original_request_result_sha256': _value_sha(json.loads(raw) if raw else None)}
        if attempt_id is not None:
            status, raw = db.execute('SELECT status,result FROM attempts WHERE id=?', (attempt_id,)).fetchone()
            overlay.update(original_attempt_status=status,
                           original_attempt_result_sha256=_value_sha(json.loads(raw) if raw else None))
        return overlay


def test_historical_recovery_overlay_releases_only_exact_verified_unsubmitted_attempt(pair):
    old, current = pair
    with sqlite3.connect(old) as db:
        db.execute("UPDATE requests SET status='unknown',result=NULL")
        db.execute("UPDATE attempts SET status='unknown',result=NULL")
        db.execute("INSERT INTO attempts VALUES(2,'receipt','unknown',NULL)")
    before = old.read_bytes()
    overlays = [recovery_overlay(old, 1)]
    cat = ReadOnlyReuseCatalog([old], method=METHOD, dataset='hotpotqa', resources=RESOURCES,
                              reconciliation_resolver=lambda path, rid: overlays)
    assert cat.lookup('q', 'final', PAYLOAD, current)['blockers']
    overlays.append(recovery_overlay(old, 2))
    value = cat.lookup('q', 'final', PAYLOAD, current)
    assert not value['blockers'] and value['candidate'] is None and len(value['historical_attempts']) == 2
    overlays[1]['evidence']['payload_sha256'] = 'changed'
    assert cat.lookup('q', 'final', PAYLOAD, current)['blockers']
    assert old.read_bytes() == before


def test_no_attempt_row_recovery_retains_conservative_initial_ordinal(pair):
    old, current = pair
    with sqlite3.connect(old) as db:
        db.execute("UPDATE requests SET status='unknown',result=NULL")
        db.execute('DELETE FROM attempts')
    overlays = [recovery_overlay(old, None)]
    cat = ReadOnlyReuseCatalog([old], method=METHOD, dataset='hotpotqa', resources=RESOURCES,
                              reconciliation_resolver=lambda path, rid: overlays)
    value = cat.lookup('q', 'final', PAYLOAD, current)
    assert not value['blockers'] and not value['attempt_count_unverified']
    assert len(value['historical_attempts']) == 1 and value['historical_attempts'][0]['attempt_id'] is None
    overlays[0]['evidence']['provider_confirmed_no_charge'] = False
    assert cat.lookup('q', 'final', PAYLOAD, current)['blockers']
