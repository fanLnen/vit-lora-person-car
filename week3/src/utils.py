import os
import torch
from peft import LoraConfig, PeftModel
from transformers import AutoImageProcessor, AutoModelForImageClassification


def compute_micro_f1(preds: torch.Tensor, labels: torch.Tensor, eps: float = 1e-8):
    """
    preds:  [N, C], values are 0/1
    labels: [N, C], values are 0/1
    """
    preds = preds.int()
    labels = labels.int()

    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return precision, recall, f1


def get_lora_config():
    return LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "key", "value", "output.dense"],
        lora_dropout=0.1,
        bias="none"
    )


def load_peft_model_and_processor(model_name: str, peft_model_dir: str):
    if not os.path.exists(peft_model_dir):
        raise FileNotFoundError(f"PEFT model directory not found: {peft_model_dir}")

    processor = AutoImageProcessor.from_pretrained(peft_model_dir)

    base_model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=2,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True
    )

    model = PeftModel.from_pretrained(base_model, peft_model_dir)
    return processor, model