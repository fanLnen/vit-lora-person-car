import os
import torch
from peft import LoraConfig, PeftModel
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


def compute_singlelabel_metrics(preds: torch.Tensor, labels: torch.Tensor, num_classes: int, eps: float = 1e-8):
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
            "f1": f1
        }

    macro_precision = sum(precision_list) / num_classes
    macro_recall = sum(recall_list) / num_classes
    macro_f1 = sum(f1_list) / num_classes

    return accuracy, macro_precision, macro_recall, macro_f1, per_class_metrics


def get_lora_config():
    return LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "key", "value", "output.dense"],
        lora_dropout=0.1,
        bias="none"
    )


def load_peft_model_and_processor(model_name: str, peft_model_dir: str, mode: str = "multilabel", num_labels: int = 2):
    if not os.path.exists(peft_model_dir):
        raise FileNotFoundError(f"PEFT model directory not found: {peft_model_dir}")

    processor = AutoImageProcessor.from_pretrained(peft_model_dir)

    problem_type = "multi_label_classification" if mode == "multilabel" else "single_label_classification"

    base_model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type=problem_type,
        ignore_mismatched_sizes=True
    )

    model = PeftModel.from_pretrained(base_model, peft_model_dir)
    return processor, model