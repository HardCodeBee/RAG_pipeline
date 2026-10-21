"""Export one recorded p02 case, without model, network or retrieval execution.

Run with python -m experiments.query_decomposition.b0_strategy_round2_v1_20261020.export_walkthrough
Frozen inputs are read only; exported prompts come directly from successful SQLite receipts.
"""
import hashlib
import json
from pathlib import Path
import sqlite3

from experiments.query_decomposition.component_chain_c08_c02_v1_20261019 import pipeline as b0
from experiments.query_decomposition.b0_strategy_round2_v1_20261020 import strategy
from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'outputs/query_decomposition/hotpot_formal_eval_v1/b0_strategy_round2_v1_20261020/fresh_dev90'
PREP = ROOT / 'experiments/query_decomposition/b0_strategy_round2_v1_20261020/fresh_validation'
OUT = ROOT / 'outputs/query_decomposition/p02_query_walkthrough_20261020'
QID = '6c7f03240bdb11eba7f7acde48001122'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def write_text(name, text):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8', newline='\n')


def write_json(name, value):
    write_text(name, json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def single_row(path, target):
    return next(v for s in path.read_text(encoding='utf-8').splitlines()
                if (v := json.loads(s)).get('id') == target)


def main():
    case_path = SOURCE / f'rounds/r000/cases/2wiki/p02_role_headers/{QID}.json'
    parent_path = SOURCE / f'rounds/r000/cases/2wiki/parent_g01/{QID}.json'
    remote_path = SOURCE / 'remote/results/027.json'
    case, parent, remote = read(case_path), read(parent_path), read(remote_path)
    identity = read(SOURCE / 'remote/results/IDENTITY.json')
    protocol = read(SOURCE / 'PROTOCOL.frozen.json')
    runtime = single_row(PREP / 'runtime_dev90.frozen.jsonl', case['id'])
    reference = single_row(PREP / 'reference_dev90.local.frozen.jsonl', case['id'])
    checks = []

    def check(name, condition):
        checks.append({'check': name, 'passed': bool(condition)})
        assert condition, name

    for label, path in [('case', case_path), ('parent', parent_path), ('remote', remote_path)]:
        dest = OUT / f'raw/{label}.json'
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        check(f'{label}: byte-identical raw copy', dest.read_bytes() == path.read_bytes())

    check('remote/query/runtime/case identity', remote['id'] == runtime['id'] == case['id']
          and remote['query'] == runtime['question'] == case['question'])
    check('remote raw parses to actual supporting questions', json.loads(remote['raw_output'])
          == remote['parsed'] == case['supporting_questions'] == runtime['supporting_questions'])
    check('remote prompt SHA', hashlib.sha256(remote['actual_prompt'].encode()).hexdigest() == identity['prompt_sha256'])
    check('remote identity SHA', digest(identity) == remote['identity_sha256'])
    check('remote greedy/no reference/truncation/error', identity['effective_do_sample'] is False
          and identity['effective_num_beams'] == 1 and not remote['explicit_references']
          and remote['format_valid'] and not remote['truncated'] and remote['generation_error'] is None)
    check('remote token ID counts', len(remote['input_token_ids']) == remote['input_tokens']
          and len(remote['output_token_ids']) == remote['output_tokens'])
    check('program appends original query as goal', [n['request'] for n in case['nodes']]
          == case['supporting_questions'] + [case['question']]
          and [n['role'] for n in case['nodes']] == ['ordinary', 'goal'])
    check('p02 equals parent upstream nodes and root', case['nodes'] == parent['nodes']
          and case['original_retrieval'] == parent['original_retrieval'])
    messages = [{'role': 'system', 'content': remote['actual_prompt']},
                {'role': 'user', 'content': case['question']}]
    write_json('00_decomposition_messages.json', messages)
    write_text('00_decomposition_prompt.txt', '[SYSTEM MESSAGE]\n' + remote['actual_prompt']
               + '\n[USER MESSAGE]\n' + case['question'])
    write_text('00_decomposition_output.txt', remote['raw_output'])
    write_json('00_decomposition_identity.json', identity)

    a, g = case['nodes']
    packets = {'01_original_query_top10': case['original_retrieval'],
               '02_child1_original_top10': a['original_retrieval'],
               '03_goal_original_top10': g['original_retrieval'],
               '04_goal_rewritten_top10': g['rewritten_retrieval'],
               '05_goal_merged_top10': g['reader_packet']}
    packet_texts = {}
    for label, packet in packets.items():
        lines = [f'Query: {packet["query"]}', f'k: {packet["k"]}']
        for d in packet['docs']:
            lines += ['', f'Rank {d["rank"]} | score={d["score"]!r} | doc_id={d["doc_id"]}', d['block']]
        packet_texts[label] = '\n'.join(lines)
        write_json(f'retrieval/{label}.json', packet)
        write_text(f'retrieval/{label}.txt', packet_texts[label])
    check('root is reused for original goal retrieval', case['original_retrieval'] == g['original_retrieval'])
    check('ordinary has no rewrite/formatter', a['rewrite_prompt'] is None and a['rewritten_retrieval'] is None
          and a['sentence_format_request_id'] is None and a['prior_complete_answers'] == [])
    check('ordinary reader uses original route only', a['reader_packet'] == a['original_retrieval'])
    check('goal rewrite accepted', g['raw_rewrite'].strip() == g['rewritten_query']
          and g['rewritten_query'] != g['request'] and g['rewritten_query'].endswith('?')
          and '\n' not in g['rewritten_query'] and g['rewrite_rejection'] is None)
    check('goal retains full previous QA', g['prior_complete_answers'] == [
        {k: a[k] for k in ['step_id', 'request', 'answer']}])
    check('goal no formatter; both sentence outputs retained', g['sentence_format_request_id'] is None
          and all(n['answer'] == n['initial_answer'] and len(n['answer'].split()) >= 4 for n in case['nodes']))
    check('merge exactly matches frozen function', b0.merge(g['original_retrieval'], g['rewritten_retrieval']) == g['reader_packet'])
    oids = [d['doc_id'] for d in g['original_retrieval']['docs']]
    rids = [d['doc_id'] for d in g['rewritten_retrieval']['docs']]
    composition = [{'reader_rank': d['rank'], 'source': d['source'], 'doc_id': d['doc_id'],
                    'route': 'original' if d['doc_id'] in oids else 'rewritten',
                    'route_rank': (oids if d['doc_id'] in oids else rids).index(d['doc_id']) + 1,
                    'score': d['score']} for d in g['reader_packet']['docs']]
    check('two routes have no common doc IDs; five from each retained', not set(oids).intersection(rids)
          and [d['route'] for d in composition] == ['original', 'rewritten'] * 5)
    write_json('retrieval/06_merge_composition.json', composition)
    check('root lacks birthplace, rewritten rank1 and merged rank2 contain it',
          'Évreux' not in b0.passages(case['original_retrieval'])
          and 'Évreux' in g['rewritten_retrieval']['docs'][0]['text']
          and 'Évreux' in g['reader_packet']['docs'][1]['text'])
    examples = read(ROOT / 'experiments/query_decomposition/hotpot_formal_eval_v1/methods/s02_child4_faithful/examples.json')
    for n in case['nodes']:
        base = b0.legacy_child.child(case['question'], n['step_id'], n['request'], n['prior_complete_answers'],
                                    b0.passages(n['reader_packet']), examples['child'], role=n['role'], combine=False)
        expected = strategy.goal_prompt(base, 'p02_role_headers') if n['role'] == 'goal' else base
        check(f'node{n["step_id"]} prompt exact renderer match', n['child_prompt'] == expected)
    check('rewrite prompt exact legacy renderer match', g['rewrite_prompt'] == b0.legacy_rewrite.rewrite(g['request'], g['prior_complete_answers']))
    expected_prompt, expected_context = strategy.render_final(case['question'], case['nodes'], case['original_retrieval'],
                                                            protocol['readout'], 'p02_role_headers')
    check('p02 helper exact renderer', strategy.helper(case['nodes'], 'p02_role_headers') == case['helper'])
    check('final prompt/context exact renderer', expected_prompt == case['final_prompt'] and expected_context == case['final_context'])
    check('final actual context is helper then original Top10 only', case['final_context']
          == case['helper'] + '\n\n' + b0.passages(case['original_retrieval'])
          and g['rewritten_retrieval']['docs'][0]['block'] not in case['final_context'])
    write_text('06_auxiliary_material.txt', case['helper'])
    write_text('07_final_context.txt', case['final_context'])

    calls = []
    names = ['01_child1_answer', '02_goal_query_rewrite', '03_goal_answer', '04_final_answer']
    expected_prompts = [a['child_prompt'], g['rewrite_prompt'], g['child_prompt'], case['final_prompt']]
    expected_answers = [a['answer'], g['rewritten_query'], g['answer'], case['prediction']]
    for logical, name, prompt, answer, limit in zip(case['logical_calls'], names, expected_prompts, expected_answers, [128, 256, 128, 64]):
        db_path = (ROOT / logical['source']).resolve()
        con = sqlite3.connect(db_path.as_uri() + '?mode=ro', uri=True)
        con.row_factory = sqlite3.Row
        row = dict(con.execute('SELECT * FROM requests WHERE id=?', (logical['request_id'],)).fetchone())
        con.close()
        payload, result = json.loads(row['payload']), json.loads(row['result'])
        check(name + ': payload SHA verified', digest(payload) == logical['payload_sha256'])
        check(name + ': full user prompt exactly matches saved case', payload['messages'] == [{'role': 'user', 'content': prompt}])
        check(name + ': generation config', payload['model'] == 'qwen-plus-2025-12-01'
              and payload['temperature'] == 0 and payload['max_tokens'] == limit
              and payload['extra_body'] == {'enable_thinking': False})
        check(name + ': successful output/usage exactly matches case receipt', row['status'] == 'success'
              and result['finish_reason'] == logical['finish_reason'] == 'stop'
              and result['answer'].strip() == answer and result['usage'] == logical['usage'])
        obj = {'logical_call': logical, 'source_database': str(db_path),
               'stored_stage': row['stage'], 'stored_status': row['status'], 'payload': payload, 'result': result}
        calls.append(obj)
        write_text(name + '_prompt.txt', prompt)
        write_text(name + '_response.txt', result['answer'])
        write_json(name + '_receipt.json', obj)
    check('four logical calls, three reused and final new', len(calls) == 4
          and [c['logical_call']['reused'] for c in calls] == [True, True, True, False])

    trace = {}
    for variant in ['parent_g01', 'p02_role_headers']:
        path = SOURCE / f'rounds/r000/methods/2wiki/{variant}/trace.jsonl'
        trace[variant] = [v for s in path.read_text(encoding='utf-8').splitlines()
                          if (v := json.loads(s)).get('qid') == QID and v.get('event') == 'logical_retrieval']
    check('four logical retrievals, three unique queries, p02 all cached', len(trace['p02_role_headers']) == 4
          and len({v['query'] for v in trace['p02_role_headers']}) == 3
          and all(v['cache_hit'] for v in trace['p02_role_headers']))
    check('parent performs three actual retrievals', [v['cache_hit'] for v in trace['parent_g01']] == [False, False, True, False])
    write_json('08_retrieval_trace.json', trace)
    scores = metric('2wiki')(case['prediction'], reference)
    check('native 2Wiki scoring', scores['f1'] == 1 and scores['em'] == 1)
    check('gold reference excluded from runtime', all(k not in runtime for k in ['answer', 'aliases', 'answer_id']))
    check('all pinned protocol sources unchanged', all(sha(ROOT / p) == h for p, h in protocol['source_pins'].items()))
    input_tokens = sum(c['result']['usage']['prompt_tokens'] for c in calls)
    output_tokens = sum(c['result']['usage']['completion_tokens'] for c in calls)
    cost = input_tokens * .8e-6 + output_tokens * 2e-6
    new_cost = sum(c['result']['usage']['prompt_tokens'] * .8e-6 + c['result']['usage']['completion_tokens'] * 2e-6
                   for c in calls if not c['logical_call']['reused'])
    accounting = {'logical_RAG_calls': 4, 'cache_reused_RAG_calls': 3, 'p02_new_RAG_calls': 1,
                  'logical_retrievals': 4, 'unique_retrieval_queries': 3, 'p02_cache_retrievals': 4,
                  'remote_decomposition_calls': 1, 'logical_RAG_input_tokens': input_tokens,
                  'logical_RAG_output_tokens': output_tokens, 'logical_full_list_price_CNY': cost,
                  'p02_new_returned_usage_price_CNY': new_cost, 'price_snapshot': {'input_CNY_per_million': .8, 'output_CNY_per_million': 2},
                  'actual_bill_verified': False, 'remote_GPU_price_included': False,
                  'p02_cached_replay_wall_seconds': case['actual_seconds'],
                  'offline_export_new_API_calls': 0, 'offline_export_new_retrievals': 0, 'offline_export_remote_calls': 0}
    verification = {'passed': all(c['passed'] for c in checks), 'checks_count': len(checks), 'checks': checks}
    write_json('verification.json', verification)
    bundle = {'scope': 'Recorded p02 walkthrough; offline export, no new inference/search/API',
              'case': case, 'parent_case': parent, 'decomposition_remote': remote,
              'decomposition_messages': messages, 'remote_identity': identity,
              'protocol': protocol, 'runtime_input': runtime, 'score_reference_NOT_generation_input': reference,
              'native_scores': scores, 'retrieval_packets': packets, 'merge_composition': composition,
              'retrieval_trace': trace, 'API_calls': calls, 'accounting': accounting,
              'verification': verification,
              'source_files': {str(p.relative_to(ROOT)): sha(p) for p in [case_path, parent_path, remote_path, SOURCE / 'PROTOCOL.frozen.json']}}
    write_json('full_walkthrough.json', bundle)

    text = [
        'p02_role_headers：一条真实 query 的完整记录',
        '本文件逐字导出已成功的提示词、材料和回答；没有重新推理、API调用、检索或远端操作。',
        '分段标记是报告边界，不是新增的模型输入。SQLite payload 和原始远端 token IDs 见 full_walkthrough.json。',
        '\n一、版本和对象',
        'p02 = legacy B0 多粒度流程 + P1最终输出要求 + g01 GOAL_RULE + 辅助问答角色标题。',
        '不是 C08/c02 整合版；不存在 S02 的临时绑定回答/正式回答双pass。',
        f'ID: {case["id"]}', f'原query: {case["question"]}',
        '标准答案仅由离线计分读取：' + reference['answer'],
        '\n二、远端4B分解输入、输出和程序追加节点',
        '本批分解先在远端预计算，之后本地执行器读取原query与supporting_questions，才开始以下RAG检索。',
        'R02基座：Qwen3-4B-Instruct-2507；v12 system prompt；greedy, do_sample=false, num_beams=1, max_new_tokens=1024, seed=1729。',
        '使用原生聊天模板；完整语义消息如下；原始输入/输出token IDs另存JSON。',
        (OUT / '00_decomposition_prompt.txt').read_text(encoding='utf8'),
        '[RAW MODEL OUTPUT]', remote['raw_output'],
        '[PROGRAM EXECUTION NODES]',
        json.dumps([{k: n[k] for k in ['step_id', 'request', 'role']} for n in case['nodes']], ensure_ascii=False, indent=2),
        '模型没有输出编号、#引用、steps/id/request/goal；最后的原query由程序追加作为goal。',
        '\n三、原query首先检索',
        '冻结全局2Wiki语料430225篇；BGE-small-en-v1.5,384维,归一化内积,无query prefix,encoder max_length512。',
        '每路Top10；无BM25、reranker、judger或按gold证据取文档。pages=1-1是本地chunk格式元数据。',
        packet_texts['01_original_query_top10'],
        '\n四、子问题1：独立原始路检索，随后一次完整句回答',
        '没有前置QA，因此不调用改写器；读入子问题1的Top10，不与原query Top10混合。',
        packet_texts['02_child1_original_top10'],
        '[COMPLETE CHILD1 USER MESSAGE]', expected_prompts[0], '[ACTUAL RESPONSE]', expected_answers[0],
        '返回6词，不触发少于4词才调用的旧完整句格式化器。该完整句直接进入前置历史和最终辅助材料。',
        '\n五、goal：先检索未改写的原query，再用前置QA改写并另路检索',
        'goal也完整执行；此题不使用retrieve/combine/skip句式路由。',
        '[COMPLETE REWRITE USER MESSAGE]', expected_prompts[1], '[ACTUAL RESPONSE]', expected_answers[1],
        '改写接受检查仅是非空、单行并以?结尾，没有语义正确性核查。当前reader问题仍保留未改写的原query。',
        '[GOAL ORIGINAL ROUTE: same packet as original query]', packet_texts['03_goal_original_top10'],
        '[GOAL REWRITTEN ROUTE]', packet_texts['04_goal_rewritten_top10'],
        '按原始rank1、改写rank1、原始rank2、改写rank2交替；按doc_id去重后保留前10，再连续编号。',
        '本题两路doc_id零重叠，最终各保留5篇。相同标题不代表相同doc_id，不进行语义去重。',
        '[MERGE COMPOSITION]', json.dumps(composition, ensure_ascii=False, indent=2),
        '[FULL MERGED READER PACKET]', packet_texts['05_goal_merged_top10'],
        '[COMPLETE GOAL USER MESSAGE]', expected_prompts[2], '[ACTUAL RESPONSE]', expected_answers[2],
        '返回11词，不触发完整句格式化。该回答保留对象、婚姻关系与出生地。',
        '\n六、组织p02辅助材料',
        case['helper'],
        'p02仅增加角色标题；QA顺序、内容不变。普通节点为Intermediate candidate results，追加goal为Candidate conclusion。',
        '\n七、最终原query回答',
        '最终Context = 上述辅助材料 + 原query原始Top10；helper置于最前，不挤掉第10篇。',
        '不直接加入改写路原文。Évreux通过goal完整句进入最终LLM；原query Top10本身没有该字符串。',
        '最终指令与4个few-shot示例均原样保留。p02实际任务的辅助材料有角色标题，但few-shot没有同步改为该角色格式。',
        '[COMPLETE FINAL USER MESSAGE]', expected_prompts[3], '[ACTUAL RESPONSE]', expected_answers[3],
        '\n八、调用接口与计费/缓存',
        '四次RAG调用均qwen-plus-2025-12-01，单条user消息，没有另加system；temperature=0,enable_thinking=false。',
        'max_tokens分别128/256/128/64。stream=true,include_usage=true是传输设置，原始响应和usage完整保留在JSON。',
        '本次p02记录中前三调用严格复用parent_g01的成功payload，final才新调用；不能把cached replay用时视为冷启动端到端延迟。',
        json.dumps(accounting, ensure_ascii=False, indent=2),
        '[LOGICAL RETRIEVAL CACHE TRACE]', json.dumps(trace, ensure_ascii=False, indent=2),
        '\n九、离线评分与边界',
        json.dumps({'prediction': case['prediction'], 'reference': reference, 'native_scores': scores}, ensure_ascii=False, indent=2),
        '该题F1=1、EM=1。它展示实际执行机制；parent_g01也答对，因此不能用这一题证明角色标题带来提升。',
        '没有显式引用的独立问题不等于执行不依赖前置回答：改写和reader仍使用完整历史。',
        '该legacy路径不使用C08的全缺答历史过滤；本题两个excluded_prior_step_ids均为空。',
        '候选标签和冲突提示不是外部事实核查保证；本题没有额外judger/LLM审查。',
        '\n十、验证', json.dumps(verification, ensure_ascii=False, indent=2),
        '完整API payload、原始响应、token IDs、所有raw case字段、来源路径、原文及doc IDs：full_walkthrough.json。',
    ]
    write_text('full_walkthrough.txt', '\n\n'.join(text) + '\n')
    print(json.dumps({'output': str(OUT), 'checks': len(checks), 'all_passed': verification['passed'],
                      'native_scores': scores, 'accounting': accounting}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
