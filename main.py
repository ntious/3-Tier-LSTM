import os
import yaml
import numpy as np
import pandas as pd

from src.load_data import load_data
from src.preprocess import (
    drop_unnecessary, encode_target, split_data,
    feature_selection_dt, scale_features, apply_smote
)
from src.plots import plot_dt_importances, plot_training_history, plot_confusion_and_roc
from src.evaluate import evaluate_scores
from src.models.lstm_model import build_hybrid_3tier_lstm
from src.explain import run_shap_explanations, run_lime_explanations

def ensure_dirs(paths: dict):
    for k in ["outputs_dir", "model_dir", "reports_dir", "figures_dir"]:
        os.makedirs(paths[k], exist_ok=True)

def reshape_for_lstm(X: np.ndarray) -> np.ndarray:
    # reshape to (samples, timesteps=1, features)
    return X.reshape((X.shape[0], 1, X.shape[1]))

def main():
    # Load config
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    np.random.seed(cfg["seed"])
    paths = cfg["paths"]
    ensure_dirs(paths)

    # 1) Load
    df = load_data(paths["data_file"])

    # 2) Preprocess
    df = drop_unnecessary(df, cfg["data"]["drop_columns"])
    df = encode_target(df, cfg["data"]["target_column"],
                       cfg["data"]["positive_label"], cfg["data"]["negative_label"])

    # Basic sanity check
    target = cfg["data"]["target_column"]
    assert target in df.columns, f"Target column '{target}' not found after preprocessing."
    feature_names_all = [c for c in df.columns if c != target]

    # 3) Split
    X_train, X_test, y_train, y_test = split_data(
        df, target_col=target,
        test_size=cfg["split"]["test_size"],
        stratify=cfg["split"]["stratify"],
        seed=cfg["seed"]
    )

    # 4) Feature Selection with Decision Tree (on original scale)
    if cfg["dt_feature_selection"]["use"]:
        top_features, dt_importances = feature_selection_dt(
            X_train, y_train,
            max_depth=cfg["dt_feature_selection"]["max_depth"],
            random_state=cfg["dt_feature_selection"]["random_state"],
            top_k=cfg["dt_feature_selection"]["top_k"]
        )
        X_train = X_train[top_features]
        X_test = X_test[top_features]
        # Plot DT importances
        dt_fig = os.path.join(paths["figures_dir"], "dt_feature_importances.png")
        plot_dt_importances(feature_names=top_features, importances=dt_importances[:len(top_features)], out_path=dt_fig)
        feature_names = top_features
    else:
        feature_names = feature_names_all

    # 5) Scaling
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test, method=cfg["scaling"]["method"])

    # 6) SMOTE on train only
    if cfg["smote"]["apply"]:
        X_train_bal, y_train_bal = apply_smote(X_train_s, y_train,
                                               k_neighbors=cfg["smote"]["k_neighbors"],
                                               seed=cfg["seed"])
    else:
        X_train_bal, y_train_bal = X_train_s, y_train.values

    # 7) Build & Train LSTM
    input_dim = X_train_bal.shape[1]
    model = build_hybrid_3tier_lstm(
        input_dim=input_dim,
        tier1_dense_units=cfg["lstm"]["tier1_dense_units"],
        lstm_units=cfg["lstm"]["lstm_units"],
        tier3_dense_units=cfg["lstm"]["tier3_dense_units"],
        dropout=cfg["lstm"]["dropout"],
        l2_val=cfg["lstm"]["l2"]
    )

    X_train_seq = X_train_bal.reshape((X_train_bal.shape[0], 1, input_dim))
    X_test_seq = X_test_s.reshape((X_test_s.shape[0], 1, input_dim))

    callbacks = [
        # EarlyStopping
        __import__("tensorflow").keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg["training"]["early_stopping_patience"],
            restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train_seq, y_train_bal,
        batch_size=cfg["training"]["batch_size"],
        epochs=cfg["training"]["epochs"],
        validation_split=cfg["training"]["validation_split"],
        callbacks=callbacks,
        verbose=2
    )

    # Save training curves
    loss_png = os.path.join(paths["figures_dir"], "training_loss.png")
    acc_png = os.path.join(paths["figures_dir"], "training_accuracy.png")
    plot_training_history(history, loss_png, acc_png)

    # 8) Evaluate
    y_score = model.predict(X_test_seq).flatten()
    metrics = evaluate_scores(y_test.values, y_score)

    # --------- CHANGED BLOCK: robust JSON saving ----------
    import json

    def _np_encoder(o):
        import numpy as np
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        return o

    # (a) Save full metrics, including the string classification report
    with open(os.path.join(paths["reports_dir"], "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=_np_encoder)

    # (b) Save numeric-only metrics (useful for dashboards)
    numeric_metrics = {}
    for k, v in metrics.items():
        if v is None:
            numeric_metrics[k] = None
        elif isinstance(v, (int, float, np.integer, np.floating)):
            numeric_metrics[k] = float(v)
    with open(os.path.join(paths["reports_dir"], "metrics_numeric.json"), "w") as f:
        json.dump(numeric_metrics, f, indent=2)

    # (c) Save the pretty classification report separately
    if "classification_report" in metrics and isinstance(metrics["classification_report"], str):
        with open(os.path.join(paths["reports_dir"], "classification_report.txt"), "w") as f:
            f.write(metrics["classification_report"])
    # --------- END CHANGED BLOCK ----------

    # Save confusion matrix and ROC
    cm_png = os.path.join(paths["figures_dir"], "confusion_matrix.png")
    roc_png = os.path.join(paths["figures_dir"], "roc_curve.png")
    plot_confusion_and_roc(y_test.values, y_score, cm_png, roc_png)

    # 9) Explainability (SHAP & LIME) — optional, can be time-consuming
    try:
        run_shap_explanations(
            model, X_train_s, X_test_s, feature_names,
            out_dir_reports=paths["reports_dir"],
            sample_bg=cfg["explainability"]["shap"]["sample_size_background"],
            sample_explain=cfg["explainability"]["shap"]["sample_size_explain"]
        )
    except Exception as e:
        print("SHAP explanation skipped due to error:", e)

    try:
        run_lime_explanations(
            model, X_train_s, X_test_s, feature_names,
            class_names=["Benign(0)", "Malignant(1)"],
            out_dir_reports=paths["reports_dir"],
            num_features=cfg["explainability"]["lime"]["num_features"],
            num_samples=cfg["explainability"]["lime"]["num_samples"]
        )
    except Exception as e:
        print("LIME explanation skipped due to error:", e)

    print("Done. Reports and figures saved in:", paths["outputs_dir"])

if __name__ == "__main__":
    main()