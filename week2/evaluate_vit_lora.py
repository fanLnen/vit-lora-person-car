import os
import ast
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification
from peft import LoraConfig, get_peft_model


# =========================
# 1. 配置
# =========================
VAL_CSV = r"D:\Praktikum_DJi\output\coco_person_car_val.csv"
IMAGE_ROOT = r"D:\Praktikum_DJi\coco"
MODEL_NAME = "google/vit-base-patch16-224"
CHECKPOINT_PATH = r"D:\Praktikum_DJi\output\checkpoints\best_vit_lora_person_car.pt"

BATCH_SIZE = 8
NUM_WORKERS = 0
THRESHOLD = 0.5
CLASS_NAMES = ["person", "car"]


# =========================
# 2. Dataset
# =========================
class CocoPersonCarMultiLabelDataset(Dataset):
    def __init__(self, csv_file, image_root, processor):
        self.df = pd.read_csv(csv_file)
        self.image_root = image_root
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        relative_path = row["file_path"]
        image_path = os.path.join(self.image_root, relative_path)

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)

        label_list = ast.literal_eval(row["label"])
        labels = torch.tensor(label_list, dtype=torch.float)

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "file_path": relative_path
        }


# =========================
# 3. 构建模型
# =========================
def build_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True
    )

    target_modules = ["query", "key", "value", "output.dense"]

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none"
    )

    model = get_peft_model(model, lora_config)

    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(state_dict)

    return processor, model


# =========================
# 4. 指标函数
# =========================
def compute_binary_metrics(preds, labels, eps=1e-8):
    preds = preds.int()
    labels = labels.int()

    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return precision, recall, f1, tp, fp, fn


def compute_micro_f1(all_preds, all_labels, eps=1e-8):
    tp = ((all_preds == 1) & (all_labels == 1)).sum().item()
    fp = ((all_preds == 1) & (all_labels == 0)).sum().item()
    fn = ((all_preds == 0) & (all_labels == 1)).sum().item()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return precision, recall, f1


# =========================
# 5. 主评估函数
# =========================
def main():
    processor, model = build_model()

    dataset = CocoPersonCarMultiLabelDataset(
        csv_file=VAL_CSV,
        image_root=IMAGE_ROOT,
        processor=processor
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values)
            probs = torch.sigmoid(outputs.logits)
            preds = (probs > THRESHOLD).int()

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu().int())

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    print("\n===== Overall Metrics =====")
    micro_p, micro_r, micro_f1 = compute_micro_f1(all_preds, all_labels)
    print(f"Micro Precision: {micro_p:.4f}")
    print(f"Micro Recall:    {micro_r:.4f}")
    print(f"Micro F1:        {micro_f1:.4f}")

    print("\n===== Per-Class Metrics =====")
    per_class_f1 = []

    for i, class_name in enumerate(CLASS_NAMES):
        p, r, f1, tp, fp, fn = compute_binary_metrics(all_preds[:, i], all_labels[:, i])
        per_class_f1.append(f1)

        print(f"\nClass: {class_name}")
        print(f"Precision: {p:.4f}")
        print(f"Recall:    {r:.4f}")
        print(f"F1:        {f1:.4f}")
        print(f"TP={tp}, FP={fp}, FN={fn}")

    macro_f1 = sum(per_class_f1) / len(per_class_f1)
    print(f"\nMacro F1: {macro_f1:.4f}")


if __name__ == "__main__":
    main()