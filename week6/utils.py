import json
import math
import os
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from peft import LoraConfig, PeftModel

try:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
except Exception:  # pragma: no cover
    accuracy_score = None
    classification_report = None
    confusion_matrix = None
    f1_score = None
    precision_score = None
    recall_score = None
    roc_auc_score = None


DEFAULT_CLASS_NAMES = ["person", "car"]


def save_json(data: Dict, json_path: str):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_markdown(text: str, md_path: str):
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(text)


def load_json(json_path: str) -> Dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def timestamp_string() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_lora_config(
    lora_r: int = 8,
    lora_alpha: int = 16,
    target_modules: Optional[List[str]] = None,
    lora_dropout: float = 0.1,
):
    if target_modules is None:
        target_modules = ["query", "key", "value", "output.dense"]

    return LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
    )


def sigmoid_tensor(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-x))


def softmax_tensor(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    x = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def count_parameters(model) -> Tuple[int, int, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / total if total > 0 else 0.0
    return trainable, total, pct


def try_get_gpu_utilization() -> Optional[float]:
    """
    Returns current GPU utilization percentage via nvidia-smi if available.
    Returns None if unavailable.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        values = [float(v.strip()) for v in result.stdout.splitlines() if v.strip()]
        if not values:
            return None
        return sum(values) / len(values)
    except Exception:
        return None


def multilabel_confusion_matrices(y_true: np.ndarray, y_pred: np.ndarray) -> List[List[List[int]]]:
    matrices = []
    num_classes = y_true.shape[1]
    for c in range(num_classes):
        yt = y_true[:, c].astype(int)
        yp = y_pred[:, c].astype(int)
        tn = int(((yt == 0) & (yp == 0)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        tp = int(((yt == 1) & (yp == 1)).sum())
        matrices.append([[tn, fp], [fn, tp]])
    return matrices


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict:
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES[: y_true.shape[1]]

    if precision_score is None:
        raise ImportError("scikit-learn is required for Week 6 evaluation metrics.")

    metrics = {
        "accuracy": float((y_true == y_pred).mean()),
        "subset_accuracy": float((y_true == y_pred).all(axis=1).mean()),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "confusion_matrix": {},
        "auc_roc_score": {},
        "per_class_metrics": {},
    }

    cms = multilabel_confusion_matrices(y_true, y_pred)

    for i, class_name in enumerate(class_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        prob = y_prob[:, i]

        precision = float(precision_score(yt, yp, zero_division=0))
        recall = float(recall_score(yt, yp, zero_division=0))
        f1 = float(f1_score(yt, yp, zero_division=0))
        acc = float(accuracy_score(yt, yp))

        try:
            auc = float(roc_auc_score(yt, prob))
        except Exception:
            auc = None

        metrics["per_class_metrics"][class_name] = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        metrics["confusion_matrix"][class_name] = cms[i]
        metrics["auc_roc_score"][class_name] = auc

    valid_aucs = [v for v in metrics["auc_roc_score"].values() if v is not None]
    metrics["auc_roc_score_macro"] = float(sum(valid_aucs) / len(valid_aucs)) if valid_aucs else None
    return metrics


def compute_singlelabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict:
    if class_names is None:
        class_names = [str(i) for i in range(y_prob.shape[1])]

    if precision_score is None:
        raise ImportError("scikit-learn is required for Week 6 evaluation metrics.")

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "auc_roc_score": None,
        "per_class_metrics": classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }

    try:
        if y_prob.shape[1] == 2:
            metrics["auc_roc_score"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            metrics["auc_roc_score"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except Exception:
        metrics["auc_roc_score"] = None

    return metrics


def plot_curve(
    x,
    y,
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: str,
    best_x: Optional[int] = None,
    best_y: Optional[float] = None,
):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o")
    if best_x is not None and best_y is not None:
        plt.scatter([best_x], [best_y], s=80)
        plt.annotate(f"best epoch={best_x}", (best_x, best_y))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_loss_accuracy_relation(val_losses, val_accuracies, best_epoch: int, save_path: str):
    plt.figure(figsize=(8, 5))
    plt.scatter(val_losses, val_accuracies)
    if 1 <= best_epoch <= len(val_losses):
        bx = val_losses[best_epoch - 1]
        by = val_accuracies[best_epoch - 1]
        plt.scatter([bx], [by], s=100)
        plt.annotate(f"best epoch={best_epoch}", (bx, by))
    plt.xlabel("Validation Loss")
    plt.ylabel("Validation Accuracy")
    plt.title("Loss vs Accuracy")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_confidence_distribution(conf_correct: List[float], conf_incorrect: List[float], save_path: str):
    plt.figure(figsize=(8, 5))
    if conf_correct:
        plt.hist(conf_correct, bins=20, alpha=0.7, label="correct")
    if conf_incorrect:
        plt.hist(conf_incorrect, bins=20, alpha=0.7, label="incorrect")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.title("Confidence Distribution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_error_type_distribution(error_counts: Dict[str, int], save_path: str):
    names = list(error_counts.keys())
    values = [error_counts[k] for k in names]
    plt.figure(figsize=(8, 5))
    plt.bar(names, values)
    plt.xlabel("Error Type")
    plt.ylabel("Count")
    plt.title("Error Type Distribution")
    plt.xticks(rotation=20)
    plt.grid(axis="y")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_single_confusion_matrix(cm: List[List[int]], class_names: List[str], title: str, save_path: str):
    arr = np.array(cm)
    plt.figure(figsize=(6, 5))
    plt.imshow(arr)
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names)
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            plt.text(j, i, int(arr[i, j]), ha="center", va="center")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_multilabel_confusion_matrices(confusion_matrices: Dict[str, List[List[int]]], save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    for class_name, cm in confusion_matrices.items():
        plot_single_confusion_matrix(cm, ["0", "1"], f"Confusion Matrix - {class_name}", os.path.join(save_dir, f"cm_{class_name}.png"))


def build_markdown_report(report: Dict) -> str:
    lines = []
    lines.append("# Final Model Evaluation Report")
    lines.append("")
    lines.append("## Experiment Summary")
    for k, v in report.get("experiment_summary", {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Hyperparameters")
    for k, v in report.get("hyperparameters", {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Performance Metrics")
    perf = report.get("performance_metrics", {})
    for k, v in perf.items():
        if isinstance(v, (dict, list)):
            lines.append(f"- {k}: `{json.dumps(v, ensure_ascii=False)}`")
        else:
            lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Training Metrics")
    for k, v in report.get("training_metrics", {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Error Analysis Summary")
    for k, v in report.get("error_analysis_summary", {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Model Information")
    for k, v in report.get("model_information", {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Hardware Utilization")
    for k, v in report.get("hardware_utilization", {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    return "\n".join(lines)
