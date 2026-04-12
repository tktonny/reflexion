"""Metric helpers for binary dementia-risk verification runs."""

from __future__ import annotations

from verification.models import VerificationCaseResult, VerificationMetrics


POSITIVE_LABEL = "cognitive_risk"
NEGATIVE_LABEL = "HC"


def compute_metrics(results: list[VerificationCaseResult]) -> VerificationMetrics:
    tp = tn = fp = fn = 0
    error_cases = 0
    skipped_cases = 0
    successful_cases = 0

    for result in results:
        if result.error:
            error_cases += 1
            continue
        if result.predicted_label is None:
            skipped_cases += 1
            continue

        successful_cases += 1
        truth_positive = result.ground_truth_label == POSITIVE_LABEL
        pred_positive = result.predicted_label == POSITIVE_LABEL

        if truth_positive and pred_positive:
            tp += 1
        elif truth_positive and not pred_positive:
            fn += 1
        elif not truth_positive and pred_positive:
            fp += 1
        else:
            tn += 1

    evaluated = tp + tn + fp + fn
    accuracy = (tp + tn) / evaluated if evaluated else None
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    specificity = tn / (tn + fp) if (tn + fp) else None
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if precision is not None and recall is not None and (precision + recall)
        else None
    )

    return VerificationMetrics(
        evaluated_cases=len(results),
        successful_cases=successful_cases,
        error_cases=error_cases,
        skipped_cases=skipped_cases,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        specificity=specificity,
        f1=f1,
        confusion_matrix={
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
        },
    )
