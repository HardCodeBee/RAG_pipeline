"""Mechanical answer extraction; never return unparsed multi-line reasoning."""
import re


def extract(text):
    explicit = list(re.finditer(r'(?im)^\s*(?:\*{1,2})?Final\s+answer\s*(?:\*{1,2})?\s*:\s*(?:\*{1,2})?\s*([^\n]+)', text))
    legacy = list(re.finditer(r'(?im)^\s*(?:So\s+)?the\s+answer\s+is\s*:?\s*([^\n]+)', text))
    matches = explicit or legacy
    if matches:
        return matches[-1].group(1).strip().strip('*').strip()
    if text.strip() and len(text.strip().splitlines()) == 1 and len(text.split()) <= 32:
        return text.strip()
    return ''
