"""Exact audited first-owner expense attribution, copied without controller state."""
def paid_query_attribution(cache_db,ledger_db,tasks,outcomes,historical,namespace):
    """Actual charges belong to the request's FIRST submitting condition.

    Include retired technical attempts; exact shared replays do not create a
    second paid owner. Unknown reservations remain visible and never dilute a
    complete-query expense window. No reads create/alter a Cache or ledger row.
    """
    task_ids={t["task_id"] for t in tasks}
    bytask={t:dict(actual_new_returned_usage_CNY=0.,unresolved_reserved_CNY=0.) for t in task_ids}
    owner={}
    for table in("requests","retired_requests"):
        for rid,stage in cache_db.execute("SELECT id,stage FROM "+table):
            matched=[t for t in task_ids if stage.startswith("full8/"+t+"/")]
            assert len(matched)==1,(table,rid,"unknown_first_owner")
            if rid in owner:assert owner[rid]==matched[0]
            owner[rid]=matched[0]
    unattributed=dict(actual_new_returned_usage_CNY=0.,unresolved_reserved_CNY=0.)
    for rid,actual,reserved in ledger_db.execute("SELECT id,actual,reserved FROM charges WHERE namespace=?",(namespace,)):
        dest=bytask[owner[rid]] if rid in owner else unattributed
        dest["actual_new_returned_usage_CNY"]+=actual if actual is not None else 0.
        dest["unresolved_reserved_CNY"]+=reserved if actual is None else 0.
    outindex={r["task_id"]:r for r in outcomes}
    result=[]
    for task in tasks:
        value=bytask[task["task_id"]]
        outcome=outindex.get(task["task_id"])
        status=outcome["status"] if outcome else "held_or_unstarted"
        baseline=historical[task["id"]]["reference_CNY"]
        eligible=(status in("success","success_truncated") and value["actual_new_returned_usage_CNY"]>0
            and value["unresolved_reserved_CNY"]==0)
        result.append({**task,**value,"status":status,"historical_complete_query_CNY":baseline,
            "eligible_for_complete_paid_query_trend":eligible,
            "historical_expense_reference_kind":historical[task["id"]]["reference_kind"],
            "attribution":"First submitting condition; complete flow with shared prefix is a same-period variant proxy, not independent cold full-cost measurement",
            "utc":outcome.get("utc") if outcome else None})
    return result,unattributed


def expense_reasons(current,all_queries):
    reasons=[]
    if current["historical_complete_query_CNY"] and current["actual_new_returned_usage_CNY"]>3*current["historical_complete_query_CNY"]:
        reasons.append("paid_query_above3x_historical_complete_query_reference")
    window=sorted([r for r in all_queries if r["eligible_for_complete_paid_query_trend"]],key=lambda r:r["utc"])[-20:]
    if len(window)==20 and sum(r["actual_new_returned_usage_CNY"]for r in window)>1.5*sum(r["historical_complete_query_CNY"]for r in window):
        reasons.append("recent20_complete_paid_queries_above1.5x_historical_references")
    return reasons

