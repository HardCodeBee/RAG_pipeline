"""Write the bounded research report after terminal outcomes are recorded."""
import json
from . import run, audit, finish_ordinary

def main():
    summary=run.report()
    assert summary['terminal_conditions']==346,'Finish only after all conditions have an outcome'
    audit.main();finish_ordinary.main()
    checked=run.read(run.OUT/'OFFLINE_AUDIT.json')
    ordinary=run.read(run.OUT/'ORDINARY_REVIEW_SUMMARY.json')
    goal_rows=[r for r in run.rows(run.OUT/'DETAILS.local.jsonl') if r['suite']=='goal']
    index={(r['arm'],r['id']):r for r in goal_rows}
    old_rows=run.rows(run.old.OUT/'DETAILS.scored.local.jsonl')
    old_index={(r['arm'],r['id']):r for r in old_rows}
    common=[i for i in summary['common_ids'] if all(old_index.get((a,i),{}).get('native_metrics',{}).get('f1') is not None for a in ('G2','I','F'))]
    inputs={r['id']:r for r in run.rows(run.HERE/'INPUTS90.frozen.jsonl')}
    history={r['id']:r for r in run.rows(run.HERE/'HISTORY90.local.frozen.jsonl')}
    refs={r['id']:r for r in run.rows(run.HERE/'REFERENCES90.local.frozen.jsonl')}
    from experiments.query_decomposition.comprehensive_eval_v1_20261015.scoring import metric
    scorers={ds:metric(ds) for ds in ('hotpotqa','2wiki','musique')}
    old_comparison=[]
    for a in ('historical_native_P02','G0','G2','I','F','GF','GO'):
        values={}
        for ds in ('all','hotpotqa','2wiki','musique'):
            ids=[i for i in common if ds=='all' or inputs[i]['dataset']==ds]
            ss=[scorers[inputs[i]['dataset']](history[i]['prediction'],refs[i]) if a=='historical_native_P02'
                else old_index[(a,i)]['native_metrics'] if a in ('G2','I','F') else index[(a,i)]['metrics'] for i in ids]
            values[ds]=dict(n=len(ids),**{k:sum(s[k] for s in ss)/len(ss) if ss else None for k in ('f1','em')})
        values['dataset_equal_mean']={k:sum(values[ds][k] for ds in ('hotpotqa','2wiki','musique'))/3 for k in ('f1','em')}
        old_comparison.append(dict(arm=a,values=values))
    run.save(run.OUT/'PREVIOUS_RESEARCH_MATCHED.json',dict(common_n=len(common),common_ids=common,table=old_comparison,
        nonconcurrent_comparison=True,all_new_inputs_equal_previous_inputs=run.sha(run.HERE/'INPUTS90.frozen.jsonl')==run.sha(run.old.HERE/'INPUTS90.frozen.jsonl')))
    lines=['# Result binding and goal operation experiments','',
        'Completed bounded side-conversation development research. Main formal evaluation remains paused. No decomposer, retrieval or production implementation was changed.','',
        '## Ordinary QA: result-use notes','',
        'D uses the descriptive E2 demonstrations. B adds two example-only notes showing how to extract the requested prior result and bind it into the current question. The actual task is identical. All76 raw ordinary outputs completed successfully.','',
        '| Cohort | Nodes/arm | D correct / wrong / pending | B correct / wrong / pending |',
        '|---|---:|---:|---:|']
    for cohort,groups in ordinary['counts'].items():
        d,b=groups['D'],groups['B']
        lines.append(f"| {cohort} | {d['n']} | {d['correct']} / {d['wrong']} / {d['pending']} | {b['correct']} / {b['wrong']} / {b['pending']} |")
    lines += ['',f"Paired correctness on37 adjudicated nodes: {ordinary['paired_stage_correct']}. Necessary-information retention: {ordinary['paired_necessary_information']}. Only{ordinary['different_text_pairs']} of38 answer texts differ.",
        'The wrong-parent case changes Liu Yuan to Liu Yin. The Araripe case changes a correct named father into a reasonable abstention: its correctness label stays positive but its useful intermediate information disappears. No correctness repair appears in this comparison.',
        '65 labels reuse exact prior answer text under identical actual-task context;11 outputs were newly reviewed locally. This is informed Codex review, not independent human gold. One malformed-entity case remains pending in both arms. Eight cases were selected stress cases. No end-to-end F1 is inferred from these ordinary-node labels.','',
        '## Goal: factual versus operation demonstrations','',
        f"All270 goal conditions have terminal records. Statuses:{summary['statuses']['goal']}. The table uses the same{summary['same_three_arm_n']} scorable questions in all three arms. Returned algorithm failures are zero-score outcomes; unresolved technical cases remain excluded.",
        'G0 preserves the original goal fact-sentence prompt at the common512-token cap. GF uses compact synthesis instructions and six fact/conclusion examples. GO adds explicit operation sentences to those same examples. All arms use the same neutral final wrapper and final64-token cap. Historical native P02 has a different native goal/final interface and is shown only as a reference.','',
        '| Dataset | Same n | Historical native F1 / EM | G0 F1 / EM | GF F1 / EM | GO F1 / EM |',
        '|---|---:|---:|---:|---:|---:|']
    for row in summary['table']:
        values=[row['dataset'],str(row['n'])]
        for a in ('historical_native_P02','G0','GF','GO'):
            s=row['scores'][a];values.append(f"{100*s['f1']:.2f} / {100*s['em']:.2f}" if s['f1'] is not None else 'unavailable')
        lines.append('| '+' | '.join(values)+' |')
    lines += ['','| Comparison | Improved / harmed / tied | F1 change (points) | Dataset-stratified95% bootstrap interval |',
        '|---|---:|---:|---:|']
    for name,p in summary['paired'].items():
        interval=checked['paired_stratified_bootstrap'].get(name,{})
        ci=interval.get('percentile95');interval_text=f"[{100*ci[0]:.2f}, {100*ci[1]:.2f}]" if ci else 'unavailable'
        delta=p['mean_F1_delta'];delta_text=f'{100*delta:.2f}' if delta is not None else 'unavailable'
        lines.append(f"| {name} | {p.get('improved',0)} / {p.get('harmed',0)} / {p.get('tied',0)} | {delta_text} | {interval_text} |")
    lines += ['', 'Bootstrap intervals use10000 paired resamples, stratified by dataset with equal dataset weights. They describe this exposed development sample, not independent confirmation. The ordinary-present84 sensitivity and output-format counts are in OFFLINE_AUDIT.json.','',
        '## Connection with the preceding research','',
        f"The following macro means use exactly the same{len(common)} questions that are scorable in both this experiment and the preceding G2/I/F experiment. The runs are nonconcurrent. All goal input records are byte-identical; native historical P02 retains its separate protocol. This prevents comparing different missing-case denominators.",'',
        '| Arm | Dataset-equal F1 | Dataset-equal EM |','|---|---:|---:|']
    for r in old_comparison:
        v=r['values']['dataset_equal_mean'];lines.append(f"| {r['arm']} | {100*v['f1']:.2f} | {100*v['em']:.2f} |")
    lines += ['','## Expense and reproducibility','',
        f"New returned-usage expense for the entire round:CNY{summary['new_returned_usage_CNY']:.6f}. Preserved unresolved reservations:CNY{summary['unresolved_reserved_CNY']:.6f}. Reservations are not confirmed expense. The audit reconciles every returned usage record to this experiment's ledger namespace.",'',
        '| Goal arm | Same successful n | Input tokens/query | Output tokens/query | Goal+final CNY/query |',
        '|---|---:|---:|---:|---:|']
    for a,v in summary['costs'].items():
        if v['n']:lines.append(f"| {a} | {v['n']} | {v['prompt_tokens']:.1f} | {v['completion_tokens']:.1f} | {v['goal_final_CNY']:.6f} |")
    lines += ['', 'These per-query figures are logical goal+final costs on the same successful cases, including any exact current-run duplicate reuse. They are not newly measured complete-RAG cost or timing. No LLM-call-count metric is added.','',
        'Frozen design and prompts are in the experiment directory. Per-condition outputs, source contexts, statement reviews, changed-final cases and the independent offline recount are retained in this output directory. No result has been promoted to the main method or published.']
    (run.OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    (run.REVIEW/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(report=str(run.OUT/'REPORT.md'),complete_conditions=346,matched_goal_cases=summary['same_three_arm_n']),ensure_ascii=True))

if __name__=='__main__':main()
