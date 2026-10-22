"""Offline-only official Hotpot answer scoring, independent from generation."""
from collections import Counter
import re
import string


def normalize_answer(s):
    s = ''.join(ch for ch in s.lower() if ch not in set(string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', s).split())


def f1_score(prediction, ground_truth):
    p, g = normalize_answer(prediction), normalize_answer(ground_truth)
    if ((p in {'yes', 'no', 'noanswer'} or g in {'yes', 'no', 'noanswer'}) and p != g):
        return 0.0, 0.0, 0.0
    pt, gt = p.split(), g.split()
    common = sum((Counter(pt) & Counter(gt)).values())
    if not common:
        return 0.0, 0.0, 0.0
    precision, recall = common / len(pt), common / len(gt)
    return 2 * precision * recall / (precision + recall), precision, recall


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)
