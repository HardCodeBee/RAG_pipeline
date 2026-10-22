"""Pure native/budget/fact-summary/local-combination prompt and parser contracts.

Only the final parsed Answer may enter the original formatter/history. Support
remains an auditable emitted candidate: a false Support makes the whole emitted
QA wrong even when its final Answer is correct. This parser validates structure,
not factual truth, completeness, a relation, or an evidence citation.
"""
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_PATH = ROOT / "src/p02_multigrain_timing_cachefix_v1/prompts/child_examples.json"
ARMS = ("N128", "N256", "F256", "R256")
CAPS = dict(N128=128, N256=256, F256=256, R256=256)
NATIVE_OUTPUT_PARAGRAPH = (
    "Return one concise factual sentence preserving the entity, relation and necessary qualifiers. "
    "If a needed fact is missing, state what cannot be determined; lack of evidence does not mean no, none, an empty set or zero. "
    "Do not add reasoning or citations."
)
TWO_FIELD_CUE = "Output the Support and Answer fields:"
SUPPORT_MODE = {
    "F256": "state only the key facts needed for the Current subquestion; do not explain a connecting operation or derive a new conclusion in this field",
    "R256": "state the key facts needed for the Current subquestion and briefly explain only the necessary entity binding, relation direction, comparison or operation connecting those facts to its answer",
}
# All four pre-existing teaching fixtures remain in their original order. Each
# pair below uses the same supplied factual propositions; R adds only the local
# operation. Their questions, prior QA, passages and final answers are unchanged.
SUPPORT_EXAMPLES = {
    "F256": (
        "The prior answer identifies the Colorado River. The passage gives its length as 1450 mi.",
        "The passage describes Clianthus flowers and gives no species count.",
        "The prior answer identifies Callicoma. The passage states it contains just one species, Callicoma serratifolia.",
        "The passage places Leycesteria's native range in temperate Asia, specifically the Himalaya and southwestern China.",
    ),
    "R256": (
        "The prior answer resolves #1 to the Colorado River; applying the passage's 1450 mi length to that same river gives the requested property.",
        "The passage describes Clianthus flowers but gives no species count; this does not establish the total, so the count remains unresolved rather than zero.",
        "Resolve #1 to Callicoma. Its only species is Callicoma serratifolia; excluding that single species leaves zero others.",
        "The passage places Leycesteria's native range in temperate Asia, specifically the Himalaya and southwestern China; use that stated native-range contrast for the current Australia question rather than evaluating the original both-genera claim.",
    ),
}
HEADER = re.compile(r"(?m)^[ \t]*(Support|Answer)[ \t]*:[ \t]*")
RESERVED_LABEL = re.compile(r"\b(?:Support|Answer)[ \t]*:")
# A colon in a natural entity/title or time is not an extra field. Reject only
# explicit competing response headings plus markup, not every colon-bearing
# line. Unknown prose still requires semantic review rather than guessed repair.
EXTRA_HEADING = re.compile(
    r"(?im)^[ \t]*(?:(?:Reasoning|Conclusion|Details|Explanation|Evidence|Final answer|Final|Analysis|Caveat|Notes?|Rationale|Response|Question|Result|Output)[ \t]*:|#{1,6}[ \t]+|```)")


class FormatFailure(ValueError):
    """Complete model output violated the two-field structure; never repair it.

    The runner records this as an algorithm-format outcome, not a retryable
    provider failure. Messages contain a reason code and never raw output.
    """
    def __init__(self, reason):
        self.reason = reason
        super().__init__("two_field_format_failure:" + reason)


def cap(arm):
    if arm not in CAPS:
        raise ValueError("Unknown local QA arm")
    return CAPS[arm]


def output_paragraph(arm):
    if arm not in SUPPORT_MODE:
        raise ValueError("Two-field instructions require F256 or R256")
    return (
        "Return exactly two labeled fields, Support: followed by Answer:, on separate lines. Do not add other headings or text.\n"
        "In Support, use at most 60 words to " + SUPPORT_MODE[arm] + ". "
        "Preserve entities, relations, conditions and units. Treat prior answers as candidate information and retain passage-supported corrections; "
        "do not introduce a factual premise that the supplied prior answers or passages do not support.\n"
        "In Answer, return one concise complete factual sentence preserving the entity, relation and necessary qualifiers. "
        "If a needed fact is missing, state what cannot be determined and preserve any established relevant partial facts; "
        "lack of evidence does not mean no, none, an empty set or zero. "
        "Answer only the Current subquestion, not the Original question. Do not add citations."
    )


def native_examples():
    values = json.loads(EXAMPLES_PATH.read_text(encoding="utf-8-sig"))
    if len(values) != 4:
        raise ValueError("Expected the frozen four native teaching fixtures")
    return values


def prompt(item, arm):
    cap(arm)
    original = item["control_prompt"]
    if arm in ("N128", "N256"):
        return original
    delimiter = "\n\nActual task:\n"
    if item.get("role", "ordinary") != "ordinary" or original.count(delimiter) != 1:
        raise ValueError("Expected one native ordinary Actual task")
    prefix, actual = original.split(delimiter, 1)
    main, example_text = prefix.split("Worked examples:\n\n", 1)
    suffix = NATIVE_OUTPUT_PARAGRAPH + "\n\n"
    if not main.endswith(suffix):
        raise ValueError("Native output paragraph mismatch")
    examples = native_examples()
    expected = "\n\n".join("Example " + str(i + 1) + "\n" + row["text"] for i, row in enumerate(examples))
    if example_text != expected:
        raise ValueError("Frozen native teaching fixtures mismatch")
    converted = []
    for index, row in enumerate(examples):
        old_tail = "\n\nAnswer: " + row["answer"]
        if not row["text"].endswith(old_tail):
            raise ValueError("Native example answer boundary mismatch")
        support = SUPPORT_EXAMPLES[arm][index]
        if len(support.split()) > 60:
            raise ValueError("Authored support fixture exceeds word guidance")
        text = row["text"][:-len(old_tail)] + "\n\nSupport: " + support + "\nAnswer: " + row["answer"]
        converted.append("Example " + str(index + 1) + "\n" + text)
    actual_tail = "\n\nAnswer:"
    if not actual.endswith(actual_tail):
        raise ValueError("Actual native answer cue mismatch")
    new_main = main[:-len(suffix)] + output_paragraph(arm) + "\n\n"
    return new_main + "Worked examples:\n\n" + "\n\n".join(converted) + delimiter + actual[:-len(actual_tail)] + "\n\n" + TWO_FIELD_CUE


def parse(arm, text):
    cap(arm)
    if not isinstance(text, str) or not text.strip():
        raise FormatFailure("empty_or_nontext")
    body = text.strip()
    if arm in ("N128", "N256"):
        return dict(support=None, answer=body)
    headers = list(HEADER.finditer(body))
    # Reserved labels may not recur inline in a value either: do not guess
    # whether a repeated Answer label was meant as a repair or a new field.
    if len(headers) != 2 or [m.group(1) for m in headers] != ["Support", "Answer"]:
        raise FormatFailure("missing_duplicate_or_out_of_order_header")
    if len(list(RESERVED_LABEL.finditer(body))) != 2 or body[:headers[0].start()].strip():
        raise FormatFailure("extra_text_or_reserved_header")
    support = body[headers[0].end():headers[1].start()].strip()
    answer = body[headers[1].end():].strip()
    if not support or not answer:
        raise FormatFailure("empty_field")
    if EXTRA_HEADING.search(support) or EXTRA_HEADING.search(answer):
        raise FormatFailure("additional_heading_or_fence")
    if re.search(r"\r?\n[ \t]*\r?\n", support) or re.search(r"\r?\n[ \t]*\r?\n", answer):
        raise FormatFailure("additional_paragraph")
    # Sixty words is prompt guidance, not an automatic factual/format penalty.
    # Keep the exact extracted field contents apart from boundary whitespace.
    return dict(support=support, answer=answer)
