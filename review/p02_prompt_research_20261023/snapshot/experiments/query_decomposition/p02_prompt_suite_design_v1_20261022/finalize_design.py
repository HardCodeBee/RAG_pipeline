"""Close only this design session after the user's minimum review duration."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import json
import re
from urllib.parse import unquote
from design_tools import HERE,INPUTS,STAGES,render


def main():
    session=json.loads((HERE/'SESSION.json').read_text(encoding='utf8'))
    now=datetime.now(timezone.utc)
    earliest=datetime.fromisoformat(session['earliest_completion_utc'])
    assert now>=earliest, 'At least one hour of review was requested; do not close early.'
    assert session['status']=='design_in_progress', 'Do not rewrite a closed session.'
    checks=json.loads((HERE/'OFFLINE_CHECKS.json').read_text(encoding='utf8'))
    assert checks['failed']==0 and checks['model_executions']==0
    english=(HERE/'revisions/r03/FULL_PROMPTS.en.md').read_text(encoding='utf8')
    chinese=(HERE/'revisions/r03/FULL_PROMPTS.zh.md').read_text(encoding='utf8')
    bank=json.loads((HERE/'revisions/r03/examples.json').read_text(encoding='utf8'))
    documentation=[]
    for stage in STAGES:
        data={k:[] if k in ('history','passages') else {'Facts':'{goal_facts}','Link':'{goal_link}',
              'Candidate':'{goal_candidate}'} if k=='synthesis' else '{'+k+'}' for k in INPUTS[stage]}
        rendered=render('r03',stage,data)
        parts=rendered.rsplit('\nActual task:\n',1)
        parts[-1]=parts[-1].replace('(No previous answers.)','{history}').replace('(No passages supplied.)','{passages}')
        assert '\nActual task:\n'.join(parts) in english,stage+' full prompt differs from actual renderer'
        assert '## '+stage in chinese
        for example in bank['stages'][stage]:assert '：'+example['id']+'\n' in chinese
        documentation.append(stage+': full English renderer matches, Chinese example coverage checked')
    links=[]
    for path in HERE.rglob('*.md'):
        for target in re.findall(r'\]\(([^\s)]+)\)',path.read_text(encoding='utf8')):
            if target.startswith(('http://','https://','#')):continue
            resolved=(path.parent/unquote(target.split('#')[0])).resolve()
            assert resolved.exists(),str(path)+': '+target
            links.append({'source':str(path.relative_to(HERE)),'target':target})
    repo=HERE.parents[2]
    for rel,wanted in session['source_core_sha256'].items():
        assert hashlib.sha256((repo/rel).read_bytes()).hexdigest()==wanted,rel+' changed since session start'
    duration=(now-datetime.fromisoformat(session['start_utc'])).total_seconds()
    session.update(status='design_complete_not_model_validated',completed_utc=now.isoformat(),
        elapsed_seconds=round(duration,2),elapsed_minutes=round(duration/60,2),
        latest_candidate='r03',prompt_revisions=['r01','r02','r03'],review_rounds=5,
        model_executions=0,local_gpu_jobs=0,remote_gpu_jobs=0,production_files_changed=False,
        validation={'structural_checks_passed':checks['passed'],'structural_checks_failed':0,
                    'authored_examples':24,'authored_challenges':40,'semantic_model_results':'not run'},
        duration_definition='Wall-clock design/review session; not CPU time or measured model latency')
    (HERE/'SESSION.json').write_text(json.dumps(session,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (HERE/'DOCUMENT_CHECKS.json').write_text(json.dumps(dict(utc=now.isoformat(),
      english_render_sync= documentation,local_links_validated=len(links),links=links,
      chinese_note='Full example/input coverage is mechanically checked; translation meaning was manually reviewed.'),
      ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    with (HERE/'WORKLOG.jsonl').open('a',encoding='utf8') as h:
        h.write(json.dumps(dict(utc=now.isoformat(),stage='design_completed',elapsed_minutes=session['elapsed_minutes'],
          review_rounds=5,latest_candidate='r03',model_executions=0),ensure_ascii=False)+'\n')
    text=f'''# P02首代新提示词套件：设计完成，尚未模型验证

本次按要求进行了至少一小时的设计与反复检查：{session['start_utc']} 至 {now.isoformat()}，共{session['elapsed_minutes']:.2f}分钟。这是实际设计会话的墙钟时间，不是模型计算时间。完整时间记录见[SESSION.json](SESSION.json)与[WORKLOG.jsonl](WORKLOG.jsonl)。

当前候选为r03。保留r01、r02、r03三个真实版本和五轮审核记录，包含4类提示词、24个虚构few-shot及40个边界检查案例。没有运行新模型评估，没有替换生产P02，主评估暂停状态保持。F1改进尚未验证。

## 建议阅读顺序

1. [设计说明及阶段交接图](DESIGN.md)：四类prompt如何配合、哪些变化超出纯文字替换。
2. [完整中文对照](revisions/r03/FULL_PROMPTS.zh.md)：全部指令、24个完整示例和动态字段。
3. [完整英文候选](revisions/r03/FULL_PROMPTS.en.md)：实际渲染出的模板正文；单独模板与examples.json在同目录。
4. [输入输出约定](revisions/r03/INTERFACE.md)：Support、Answer、goal三字段、材料与异常如何处理。
5. [后续验证顺序](VALIDATION_PLAN.md)：上游/下游四组比较、历史与同期控制区别、完整动态流程与开发样本边界。
6. [已有实验依据](EVIDENCE.md)、[源记录索引](SOURCE_INDEX.md)：设计动机及证据局限。

## 五轮实质审核

- [初稿问题](reviews/01_initial_critique.md)：parent被误推为father、final示例多拿了goal材料、只有弃答缺少充分下界等。
- [阶段语义与接口](reviews/02_semantics_and_interfaces.md)：goal示例引用未传入的亲属事实、部分列表有用性、原题对象回映射等。
- [真实失效案例对照](reviews/03_real_case_transfer.md)：对应刘娥、运河、职位、错误家庭等案例，并列明prompt无法补回的检索缺口。
- [反向质疑与能力边界](reviews/04_limits_and_counterarguments.md)：新字段锚定、解释错误、信息压缩与解析局限。
- [最终充分性检查](reviews/05_final_semantic_pass.md)：修复C37中“两个已知任职年份不足以证明第一份公职”的材料问题，核对历史控制差异。

## 检查结果

[194项离线结构与接线检查](OFFLINE_CHECKS.json)通过，覆盖原题保留、字段交接、审核信息隔离、异常响应及受保护源文件未改动。[文档检查](DOCUMENT_CHECKS.json)确认英文全文与渲染器一致、本地链接可读取，并核对中文全部示例覆盖。

[40个合成案例](CHALLENGE_REVIEW.md)的预期由人工编写，尚未请求模型作答；不存在40/40模型通过率。24个示例是教学材料，同样不属于实验成果。完整英文格式是候选运行内容，中文仅供审阅。

本轮形成可审阅、可接入后续隔离试验的设计包；未声称它已提升F1或构成新型提示方法。
'''
    (HERE/'README.md').write_text(text,encoding='utf8')
    print(json.dumps({k:session[k] for k in ('status','completed_utc','elapsed_minutes','latest_candidate','review_rounds','model_executions')},ensure_ascii=False))


if __name__=='__main__':main()
