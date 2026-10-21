"""Freeze question-only reference plans before retrieval/answer generation."""

from .prepare import HERE, OUT, PILOT_SPEC, rows, load, write


def main():
    target = HERE / "cases.frozen.jsonl"
    if target.exists():
        print("Preserve already frozen reference plans")
        return
    drafts = load(HERE / "reference_intermediates.json")
    cases = []
    for case in rows(HERE / "new36.questions.frozen.jsonl"):
        requests = drafts[case["qid"]] + [case["question"]]
        assert all("#" not in request for request in requests)
        cases.append({**case, "qdmr": {"steps": [{"id": i, "request": q} for i, q in enumerate(requests, 1)], "goal": len(requests)}, "plan_provenance": "Current Codex question-only reference plan, reviewed before answer generation; no intermediate answers or support annotations inserted; not independent human gold"})
    assert len(cases) == 36
    target.write_text("".join(__import__("json").dumps(r, ensure_ascii=False) + "\n" for r in cases), encoding="utf-8")
    write(HERE / "REFERENCE_REVIEW.json", {
        "review": "current Codex context reviewed36 draft plans against the original questions before any expansion candidate calls",
        "criteria": ["self-contained requests without forced references", "preserve original goal verbatim", "preserve supplied scopes, dates and relation directions", "no factual unknown answers supplied", "no fixed minimum-hop requirement"],
        "independent_human_gold": False,
        "known_question_ambiguities_preserved": {
            "2hop__156085_436709": "Possessive attachment in creator of Mickey Mouse's spouse is ambiguous. Retrieve creator information and Mickey spouse information independently; goal remains verbatim rather than imposing an answer-derived interpretation.",
            "4hop2__567956_14670_8987_8529": "Normalization/invasion role wording is difficult; intermediate wording preserves the stated direction, goal unchanged, no gold-driven correction.",
            "4hop2__71753_648517_70784_79935": "Geographic relative-clause scope is ambiguous; retrieve component descriptors independently and preserve original goal.",
            "5a8333115542993344745feb": "Helena may name multiple county seats; no country/county assumption inserted."
        },
        "limit": "The intended ideal-decomposition assumption is not established by reference drafting; ambiguous source questions remain explicit limitations rather than being silently removed."
    })
    write(HERE / "COHORT60_FROZEN.json", {"old_cases_source": str((PILOT_SPEC / "cases.frozen.jsonl").resolve()), "old_count": 24, "new_cases_source": str(target.resolve()), "new_count": 36, "total": 60, "old_case_outputs_rerun": False})
    print("Frozen36 reference plans;60-case cohort")


if __name__ == "__main__":
    main()
