import numpy as np
import pandas as pd

from src.evaluate import evaluate_scores
from src.preprocess import (
    apply_smote,
    drop_unnecessary,
    encode_target,
    scale_features,
    split_data,
)


def make_dataset():
    return pd.DataFrame(
        {
            "id": np.arange(20),
            "feature_a": np.arange(20, dtype=float),
            "feature_b": np.tile([0.0, 1.0], 10),
            "diagnosis": ["B", "M"] * 10,
        }
    )


def test_preprocessing_pipeline_is_deterministic_and_balanced():
    data = encode_target(drop_unnecessary(make_dataset(), ["id"]), "diagnosis")
    X_train, X_test, y_train, y_test = split_data(data, "diagnosis", seed=7)
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)
    X_resampled, y_resampled = apply_smote(
        X_train_scaled,
        y_train,
        k_neighbors=2,
        seed=7,
    )

    assert X_train_scaled.shape[1] == 2
    assert X_test_scaled.shape == (len(y_test), 2)
    assert len(X_resampled) == len(y_resampled)
    assert np.bincount(np.asarray(y_resampled)).tolist() == [8, 8]


def test_evaluate_scores_handles_probabilities_and_hard_labels():
    probability_metrics = evaluate_scores(
        [0, 1, 0, 1],
        [0.1, 0.9, 0.2, 0.8],
    )
    label_metrics = evaluate_scores([0, 1, 0, 1], [0, 1, 0, 1])

    assert probability_metrics["accuracy"] == 1.0
    assert probability_metrics["auc"] == 1.0
    assert label_metrics["accuracy"] == 1.0
    assert label_metrics["auc"] is None
