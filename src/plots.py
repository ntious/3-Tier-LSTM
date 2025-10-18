import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def plot_dt_importances(feature_names, importances, out_path: str):
    ensure_dir(os.path.dirname(out_path))
    order = np.argsort(importances)[::-1]
    names_sorted = np.array(feature_names)[order]
    imps_sorted = np.array(importances)[order]

    plt.figure()
    plt.barh(names_sorted, imps_sorted)
    plt.gca().invert_yaxis()
    plt.title("Decision Tree Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_training_history(history, out_path_loss: str, out_path_acc: str):
    ensure_dir(os.path.dirname(out_path_loss))
    # Loss
    plt.figure()
    plt.plot(history.history["loss"], label="loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val_loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path_loss, dpi=160)
    plt.close()

    # Accuracy
    plt.figure()
    plt.plot(history.history["accuracy"], label="accuracy")
    if "val_accuracy" in history.history:
        plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path_acc, dpi=160)
    plt.close()

def plot_confusion_and_roc(y_true, y_score, out_cm: str, out_roc: str):
    ensure_dir(os.path.dirname(out_cm))
    y_pred = (y_score >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    plt.figure()
    disp.plot(values_format='d')
    plt.title("Confusion Matrix (0=Benign, 1=Malignant)")
    plt.tight_layout()
    plt.savefig(out_cm, dpi=160)
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_roc, dpi=160)
    plt.close()
