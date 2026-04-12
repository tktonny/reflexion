from verification.metrics import compute_metrics
from verification.models import VerificationCaseResult


def test_compute_metrics_binary_counts() -> None:
    metrics = compute_metrics(
        [
            VerificationCaseResult(
                case_id="case-a",
                dataset="demo",
                split="test",
                ground_truth_label="cognitive_risk",
                predicted_label="cognitive_risk",
                transcript_model="asr",
                classifier_model="cls",
            ),
            VerificationCaseResult(
                case_id="case-b",
                dataset="demo",
                split="test",
                ground_truth_label="HC",
                predicted_label="HC",
                transcript_model="asr",
                classifier_model="cls",
            ),
            VerificationCaseResult(
                case_id="case-c",
                dataset="demo",
                split="test",
                ground_truth_label="HC",
                predicted_label="cognitive_risk",
                transcript_model="asr",
                classifier_model="cls",
            ),
            VerificationCaseResult(
                case_id="case-d",
                dataset="demo",
                split="test",
                ground_truth_label="cognitive_risk",
                predicted_label="HC",
                transcript_model="asr",
                classifier_model="cls",
            ),
        ]
    )

    assert metrics.successful_cases == 4
    assert metrics.error_cases == 0
    assert metrics.confusion_matrix == {
        "true_positive": 1,
        "true_negative": 1,
        "false_positive": 1,
        "false_negative": 1,
    }
    assert metrics.accuracy == 0.5
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.specificity == 0.5
    assert metrics.f1 == 0.5
