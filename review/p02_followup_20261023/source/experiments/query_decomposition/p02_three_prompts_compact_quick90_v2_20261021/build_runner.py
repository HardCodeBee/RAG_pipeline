"""Create an isolated variant of the already exercised quick90 runner."""
from pathlib import Path
import ast

here = Path(__file__).resolve().parent
old = here.parent / 'p02_three_prompts_quick90_v1_20261021'
code = (old / 'run.py').read_text(encoding='utf8')
code = code.replace('import sys\n', 'import sys\nimport time\n')
code = code.replace('from experiments.query_decomposition.p02_childprompt_selfcontained_quick90_v1_20261021.run_r004 import RecoveryAsk', '''from experiments.query_decomposition.p02_childprompt_selfcontained_quick90_v1_20261021.run_r002 import RecordingClient, RecordingLLM
from p02_multigrain_timing_cachefix_v1.core.runtime import Ask

SUBMIT_LOCK = Lock()
LAST_START = 0.0

class PacedLLM(RecordingLLM):
    def request_once(self, payload):
        global LAST_START
        with SUBMIT_LOCK:
            remaining = 3.0 - (time.monotonic() - LAST_START)
            if remaining > 0:
                time.sleep(remaining)
            LAST_START = time.monotonic()
            return super().request_once(payload)

class RecoveryAsk(Ask):
    def __init__(self, *args, **kwargs):
        import httpx
        from openai import OpenAI
        super().__init__(*args, **kwargs)
        timeout = httpx.Timeout(self.settings['timeout_seconds'], connect=10)
        transport = httpx.HTTPTransport(retries=2,
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=0))
        http = httpx.Client(transport=transport, timeout=timeout, trust_env=False)
        api = OpenAI(api_key=os.environ[self.settings['api_key_env']],
            base_url=self.settings['base_url'], timeout=timeout, max_retries=0, http_client=http)
        self.client = PacedLLM(self.settings, client=RecordingClient(api))
        self.client.owns_client = True
''')
code = code.replace("METHOD = 'p02_three_prompts_user_v1'", "METHOD = 'p02_three_prompts_user_compact_v2'")
start = code.index("    assert changed.count('context = block")
end = code.index('    ast.parse(changed)', start)
code = code[:start] + '''    first = changed.index('def render_final(')
    last = changed.index('\\n\\nclass Pipeline:', first)
    changed = changed[:first] + """def render_final(question, nodes, root, readout, examples):
    return final_prompts.render_final(question, nodes, passages(root))
""" + changed[last:]
''' + code[end:]
code = code.replace("'authority': 'User: 在90题快速测试集上 我们先整体评估三段'", "'authority': 'User: 使用这一套再试试 (the three compact prompts supplied on 2026-10-21)'")
code = code.replace("'failure_policy': 'Existing connect-only retries2, SDK POST retries0.", "'failure_policy': 'Provider concurrency1, minimum start interval3sec, no keepalive; connect-only retries2, SDK POST retries0.")
code = code.replace("prior_dirs = [common.OUT / 'approved_recovery004',", "prior_dirs = [OUT.parent / 'p02_three_prompts_quick90_v1_20261021/approved_unknown_recovery003', common.OUT / 'approved_recovery004',")
code = code.replace("assert 'Current subquestion #' not in actual", "assert ('Current subquestion #' + str(node['step_id']) + ':') in actual")
code = code.replace("'GOAL_RETRIEVED_PASSAGES':common.baseline.passages(node['reader_packet'])}", "'GOAL_RETRIEVED_PASSAGES':common.baseline.passages(node['reader_packet']),'GOAL_STEP_ID':node['step_id']}")
code = code.replace("            if nodes[:-1]:\n                assert final == prompts.FINAL.format_map(values)", "            assert final == prompts.FINAL.format_map(values)")
code = code.replace("final.count('Example ') == 5", "final.count('Example ') == 4")
code = code.replace("assert context == block + '\\n\\nRetrieved passages:\\n' + common.baseline.passages(packet)", "assert context == block + '\\n\\n' + common.baseline.passages(packet)")
code = code.replace("        'exact_user_template_rendering_verified':True", "        'zero_ordinary_cases_checked':sum(not c['supporting_questions'] for c in rows(COHORT/'runtime90.frozen.jsonl')),\n        'exact_user_template_rendering_verified':True")
ast.parse(code)
(here / 'run.py').write_text(code, encoding='utf8')
(here / 'analyze.py').write_text((old / 'analyze.py').read_text(encoding='utf8'), encoding='utf8')
print('Created isolated runner; original files unchanged')
