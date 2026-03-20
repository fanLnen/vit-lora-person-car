import os
import ast
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification
from peft import LoraConfig, get_peft_model



# 1. 配置区

TRAIN_CSV = r"D:\Praktikum_DJi\output\coco_person_car_train.csv"
VAL_CSV = r"D:\Praktikum_DJi\output\coco_person_car_val.csv"
IMAGE_ROOT = r"D:\Praktikum_DJi\coco"

MODEL_NAME = "google/vit-base-patch16-224"

BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-4
THRESHOLD = 0.5
NUM_WORKERS = 0  
SAVE_DIR = r"D:\Praktikum_DJi\output\checkpoints"

os.makedirs(SAVE_DIR, exist_ok=True)



# 2. Dataset

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
            "labels": labels
        }



# 3. 指标函数

def compute_micro_f1(preds: torch.Tensor, labels: torch.Tensor, eps: float = 1e-8):
    """
    preds:  [N, 2]，0/1
    labels: [N, 2]，0/1
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



# 4. 训练一个 epoch

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        probs = torch.sigmoid(logits)
        preds = (probs > THRESHOLD).int()

        all_preds.append(preds.detach().cpu())
        all_labels.append(labels.detach().cpu().int())

        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / len(dataloader)

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    precision, recall, f1 = compute_micro_f1(all_preds, all_labels)

    return avg_loss, precision, recall, f1



# 5. 验证一个 epoch

def validate_one_epoch(model, dataloader, device):
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Validation", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            running_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > THRESHOLD).int()

            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu().int())

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / len(dataloader)

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    precision, recall, f1 = compute_micro_f1(all_preds, all_labels)

    return avg_loss, precision, recall, f1



# 6. 主函数

def main():
    print("===== Step 1: Load Processor =====")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    print("===== Step 2: Load Base Model =====")
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True
    )

    print("===== Step 3: Inject LoRA =====")
    target_modules = ["query", "key", "value", "output.dense"]

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("===== Step 4: Prepare Dataset =====")
    train_dataset = CocoPersonCarMultiLabelDataset(
        csv_file=TRAIN_CSV,
        image_root=IMAGE_ROOT,
        processor=processor
    )

    val_dataset = CocoPersonCarMultiLabelDataset(
        csv_file=VAL_CSV,
        image_root=IMAGE_ROOT,
        processor=processor
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    print("===== Step 5: Prepare DataLoader =====")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"===== Step 6: Using device: {device} =====")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_val_f1 = -1.0

    print("===== Step 7: Start Training =====")
    for epoch in range(EPOCHS):
        print(f"\n========== Epoch {epoch + 1}/{EPOCHS} ==========")

        train_loss, train_precision, train_recall, train_f1 = train_one_epoch(
            model, train_loader, optimizer, device
        )

        val_loss, val_precision, val_recall, val_f1 = validate_one_epoch(
            model, val_loader, device
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Precision: {train_precision:.4f} | "
            f"Train Recall: {train_recall:.4f} | "
            f"Train F1: {train_f1:.4f}"
        )

        print(
            f"Val   Loss: {val_loss:.4f} | "
            f"Val   Precision: {val_precision:.4f} | "
            f"Val   Recall: {val_recall:.4f} | "
            f"Val   F1: {val_f1:.4f}"
        )

        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_path = os.path.join(SAVE_DIR, "best_vit_lora_person_car.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to: {best_model_path}")

    # 保存最后一轮
    last_model_path = os.path.join(SAVE_DIR, "last_vit_lora_person_car.pt")
    torch.save(model.state_dict(), last_model_path)
    print(f"\nLast model saved to: {last_model_path}")
    print(f"Best Val F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()