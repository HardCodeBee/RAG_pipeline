"""Meaningful local contracts, without Cache, provider, credentials or gold."""
from pathlib import Path
import hashlib
import json
from . import templates as t
from experiments.query_decomposition.p02_reference_binding_priority_probe_v1_20261022.templates import formatter_prompt, needs_formatter

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "p02_reference_binding_priority_probe_v1_20261022/INPUTS.frozen.jsonl"


def check():
    items = [json.loads(s) for s in SOURCE.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    assert len(items) == 38
    examples = t.native_examples(); converted_count = 0
    for item in items:
        original = item["control_prompt"]
        assert t.prompt(item, "N128") == t.prompt(item, "N256") == original
        prefix, actual = original.split("\n\nActual task:\n", 1)
        first_lines = prefix.split(t.NATIVE_OUTPUT_PARAGRAPH, 1)[0]
        for arm in ("F256", "R256"):
            result = t.prompt(item, arm)
            changed_prefix, changed_actual = result.split("\n\nActual task:\n", 1)
            assert changed_prefix.startswith(first_lines)
            assert changed_actual == actual[:-len("\n\nAnswer:")] + "\n\n" + t.TWO_FIELD_CUE
            assert actual.endswith("\n\nAnswer:")
            rendered = changed_prefix.split("Worked examples:\n\n", 1)[1]
            for index, example in enumerate(examples):
                old_tail = "\n\nAnswer: " + example["answer"]
                unchanged = example["text"][:-len(old_tail)]
                expected = unchanged + "\n\nSupport: " + t.SUPPORT_EXAMPLES[arm][index] + "\nAnswer: " + example["answer"]
                assert expected in rendered
                assert len(t.SUPPORT_EXAMPLES[arm][index].split()) <= 60
                parsed = t.parse(arm, "Support: " + t.SUPPORT_EXAMPLES[arm][index] + "\nAnswer: " + example["answer"])
                assert parsed["answer"] == example["answer"]
                converted_count += 1
        assert t.cap("N128") == 128 and all(t.cap(a) == 256 for a in t.ARMS[1:])
    for arm in t.ARMS:
        if arm.startswith("N"):
            assert t.parse(arm, "  A full fact.\n") == dict(support=None, answer="A full fact.")
            assert t.parse(arm, "Answer: native literal") == dict(support=None, answer="Answer: native literal")
        else:
            assert t.parse(arm, " \r\n Support :  One supplied fact.\r\n\r\n Answer :  The requested complete fact. \r\n") == dict(support="One supplied fact.", answer="The requested complete fact.")
            assert t.parse(arm, "Support:\nA supplied fact\nwith line wrapping.\nAnswer:\nA complete fact.")["answer"] == "A complete fact."
            assert t.parse(arm, "Support: " + "word " * 61 + "\nAnswer: A complete fact.")["support"]
            colon_output = "Support: Star Wars: Episode IV is the named film; its screening starts at 10:30.\nAnswer: Star Wars: Episode IV starts at 10:30."
            assert t.parse(arm, colon_output) == dict(
                support="Star Wars: Episode IV is the named film; its screening starts at 10:30.",
                answer="Star Wars: Episode IV starts at 10:30.")
            for extra in ("Reasoning:", "Conclusion:", "Details:"):
                try: t.parse(arm, "Support: supplied facts.\nAnswer: a complete fact.\n" + extra + " extra block.")
                except t.FormatFailure: pass
                else: raise AssertionError("Explicit extra heading accepted")
            bad = ("A plain unlabelled fact.", "Answer: fact.\nSupport: facts.",
                "Support: facts.\nAnswer: first.\nAnswer: second.",
                "Support: facts. Answer: inline.\nAnswer: fact.",
                "Preface\nSupport: facts.\nAnswer: fact.", "Support:\nAnswer: fact.",
                "Support: facts.\nAnswer:", "Support: facts.\nAnswer: fact.\nConclusion: extra.",
                "Support: facts.\nAnswer: fact.\n\nExtra paragraph.",
                "```\nSupport: facts.\nAnswer: fact.\n```", "support: fact.\nAnswer: fact.")
            for text in bad:
                try: t.parse(arm, text)
                except t.FormatFailure as exc:
                    assert isinstance(exc, ValueError) and text not in str(exc)
                else: raise AssertionError("Malformed structure accepted")
        # Fake downstream consumer uses the already-native formatter template,
        # with the parsed Answer only. No prompt/client/model call is made.
        raw = "one two three" if arm.startswith("N") else "Support: UNIQUE_SUPPORT_DO_NOT_FORWARD.\nAnswer: one two three"
        parsed = t.parse(arm, raw)
        assert needs_formatter(parsed["answer"])
        formatted_prompt = formatter_prompt(items[0], parsed["answer"])
        assert "UNIQUE_SUPPORT_DO_NOT_FORWARD" not in formatted_prompt
        assert "Question:\n" + items[0]["current_question"] + "\n\n" in formatted_prompt
        assert "Candidate answer:\none two three\n\nContext:\n" in formatted_prompt
        assert formatted_prompt.endswith("\n\n".join(d["block"] for d in items[0]["reader_packet"]["docs"]))
    for invalid in (None, "", "  "):
        for arm in t.ARMS:
            try: t.parse(arm, invalid)
            except t.FormatFailure: pass
            else: raise AssertionError("Empty output accepted")
    result = dict(passed=True, frozen_nodes_checked=38, arms=list(t.ARMS),
        native_controls_prompt_identical=True, native_cap_only_128_vs256=True,
        F_R_actual_question_history_reader_bytes_preserved=True,
        native_scope_reference_and_prior_correction_prefix_identical=True,
        original_four_example_inputs_and_answers_preserved=True, converted_example_contracts=converted_count,
        positive_whitespace_and_linewrap_parser_contracts=True,
        negative_missing_reordered_duplicate_extra_heading_fence_empty_contracts=True,
        colon_proper_titles_and_time_retained=True,
        sixty_word_guidance_not_factual_or_format_failure=True,
        native_formatter_external_extracted_Answer_only=True,
        no_fact_or_relation_verification_claimed_by_parser=True,
        source_inputs_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        templates_sha256=hashlib.sha256((HERE / "templates.py").read_bytes()).hexdigest(),
        API_SSH_GPU_requests=0, live_checkpoint_or_ledger_mutations=0)
    target = HERE / "TEMPLATE_CONTRACTS.json"
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if target.exists():
        assert target.read_text(encoding="utf8") == encoded, "Keep earlier contract result or explicitly version it"
    else:
        target.write_text(encoded, encoding="utf8")
    return result


if __name__ == "__main__": print(json.dumps(check(), ensure_ascii=True))
