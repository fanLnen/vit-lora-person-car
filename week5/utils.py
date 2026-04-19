import json
import os
from typing import Dict, List, Optional, Tuple

import torch
from peft import LoraConfig, PeftModel, TaskType
from transformers import AutoImageProcessor, AutoModelForImageClassification


def compute_multilabel_metrics(preds: torch.Tensor, labels: torch.Tensor, eps: float = 1e-8):
    """
    preds:  [N, C], values are 0/1
    labels: [N, C], values are 0/1

    Returns:
        accuracy : label-wise accuracy
        precision
        recall
        f1
    """
    preds = preds.int()
    labels = labels.int()

    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    accuracy = (preds == labels).float().mean().item()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return accuracy, precision, recall, f1


def compute_singlelabel_metrics(
    preds: torch.Tensor, labels: torch.Tensor, num_classes: int, eps: float = 1e-8
):
    """
    preds:  [N], predicted class indices
    labels: [N], ground-truth class indices

    Returns:
        accuracy
        macro_precision
        macro_recall
        macro_f1
        per_class_metrics: dict
    """
    preds = preds.long()
    labels = labels.long()

    accuracy = (preds == labels).float().mean().item()

    precision_list = []
    recall_list = []
    f1_list = []
    per_class_metrics = {}

    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

        per_class_metrics[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    macro_precision = sum(precision_list) / num_classes
    macro_recall = sum(recall_list) / num_classes
    macro_f1 = sum(f1_list) / num_classes

    return accuracy, macro_precision, macro_recall, macro_f1, per_class_metrics


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


def load_peft_model_and_processor(
    model_name: str, peft_model_dir: str, mode: str = "multilabel", num_labels: int = 2
):
    if not os.path.exists(peft_model_dir):
        raise FileNotFoundError(f"PEFT model directory not found: {peft_model_dir}")

    processor = AutoImageProcessor.from_pretrained(peft_model_dir)

    problem_type = (
        "multi_label_classification" if mode == "multilabel" else "single_label_classification"
    )

    base_model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type=problem_type,
        ignore_mismatched_sizes=True,
    )

    model = PeftModel.from_pretrained(base_model, peft_model_dir)
    return processor, model


def save_json(data: Dict, json_path: str):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
