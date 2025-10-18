import os
import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def run_shap_explanations(model, X_train_scaled, X_test_scaled, feature_names, out_dir_reports, sample_bg=100, sample_explain=50):
    ensure_dir(out_dir_reports)
    # Sample background and explain subsets for runtime friendliness
    bg_idx = np.random.choice(X_train_scaled.shape[0], size=min(sample_bg, X_train_scaled.shape[0]), replace=False)
    exp_idx = np.random.choice(X_test_scaled.shape[0], size=min(sample_explain, X_test_scaled.shape[0]), replace=False)

    background = X_train_scaled[bg_idx]
    explain_X = X_test_scaled[exp_idx]

    # Model expects shape (n, 1, features)
    def model_predict(x):
        return model.predict(x.reshape((-1, 1, x.shape[1]))).flatten()

    explainer = shap.KernelExplainer(model_predict, background)
    shap_values = explainer.shap_values(explain_X, nsamples=100)  # adjust for runtime
    np.save(os.path.join(out_dir_reports, "shap_values.npy"), shap_values)
    np.save(os.path.join(out_dir_reports, "shap_explain_X.npy"), explain_X)
    with open(os.path.join(out_dir_reports, "shap_features.txt"), "w") as f:
        for n in feature_names:
            f.write(f"{n}\n")

def run_lime_explanations(model, X_train_scaled, X_test_scaled, feature_names, class_names, out_dir_reports, num_features=10, num_samples=5000):
    ensure_dir(out_dir_reports)

    explainer = LimeTabularExplainer(
        training_data=X_train_scaled,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True
    )

    # Explain a few samples
    num_instances = min(5, X_test_scaled.shape[0])
    for i in range(num_instances):
        x = X_test_scaled[i]
        def predict_fn(xa):
            # model expects (n, 1, features)
            return np.column_stack([1 - model.predict(xa.reshape((-1, 1, xa.shape[1]))).flatten(),
                                    model.predict(xa.reshape((-1, 1, xa.shape[1]))).flatten()])
        exp = explainer.explain_instance(x, predict_fn, num_features=num_features, num_samples=num_samples)
        html_path = os.path.join(out_dir_reports, f"lime_explanation_{i}.html")
        exp.save_to_file(html_path)
