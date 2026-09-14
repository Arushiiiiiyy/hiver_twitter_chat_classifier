"""Automated metrics computed over golden_set.jsonl + predictions.jsonl."""
from __future__ import annotations

from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score


def intent_accuracy(gold: list[str], pred: list[str]) -> dict:
    return {
        "accuracy": accuracy_score(gold, pred),
        "macro_f1": f1_score(gold, pred, average="macro", zero_division=0),
    }


def escalation_agreement(gold: list[bool], pred: list[bool]) -> dict:
    return {
        "accuracy": accuracy_score(gold, pred),
        # Kappa matters more than accuracy here: if most examples aren't escalated,
        # a system that never escalates scores high accuracy but zero real skill.
        "cohen_kappa": cohen_kappa_score(gold, pred),
    }


def judge_human_agreement(judge_scores: list[int], human_scores: list[int]) -> dict:
    """Agreement between the LLM judge and a human on the same reply-quality scale.
    Report this explicitly in the report — a judge that doesn't correlate with human
    judgment invalidates every downstream quality number that relies on it."""
    exact_match = sum(j == h for j, h in zip(judge_scores, human_scores)) / len(judge_scores)
    within_one = sum(abs(j - h) <= 1 for j, h in zip(judge_scores, human_scores)) / len(judge_scores)
    return {
        "exact_match_rate": exact_match,
        "within_one_point_rate": within_one,
        "cohen_kappa": cohen_kappa_score(judge_scores, human_scores),
    }
