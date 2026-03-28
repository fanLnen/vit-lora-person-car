import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    get_linear_schedule_with_warmup,
)
from peft import get_peft_model

from dataset import CocoPersonCarMultiLabelDataset
from utils import compute_micro_f1, get_lora_config


# =========================
# 1. Config
# =========================
TRAIN_CSV = r"D:\Praktikum_DJi\output\coco_person_car_train.csv"
VAL_CSV = r"D:\Praktikum_DJi\output\coco_person_car_val.csv"
IMAGE_ROOT = r"D:\Praktikum_DJi\coco"

MODEL_NAME = "google/vit-base-patch16-224"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "models")
LOG_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "logs")

PEFT_MODEL_ID = os.path.join(MODEL_OUTPUT_DIR, "vit-base-patch16-224-lora-person-car")

BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
THRESHOLD = 0.5
NUM_WORKERS = 0
WARMUP_RATIO = 0.1

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_OUTPUT_DIR, exist_ok=True)
os.makedirs(PEFT_MODEL_ID, exist_ok=True)


# =========================
# 2. Train / Validation
# =========================
def train_one_epoch(model, dataloader, optimizer, criterion, scheduler, accelerator):
    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        optimizer.zero_grad()

        outputs = model(pixel_values=batch["pixel_values"])
        logits = outputs.logits
        loss = criterion(logits, batch["labels"])

        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        probs = torch.sigmoid(logits)
        preds = (probs > THRESHOLD).int()

        gathered_preds = accelerator.gather(preds).detach().cpu()
        gathered_labels = accelerator.gather(batch["labels"]).detach().cpu().int()

        all_preds.append(gathered_preds)
        all_labels.append(gathered_labels)

        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / len(dataloader)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    precision, recall, f1 = compute_micro_f1(all_preds, all_labels)
    return avg_loss, precision, recall, f1


@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, accelerator):
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Validation", leave=False)

    for batch in progress_bar:
        outputs = model(pixel_values=batch["pixel_values"])
        logits = outputs.logits
        loss = criterion(logits, batch["labels"])

        running_loss += loss.item()

        probs = torch.sigmoid(logits)
        preds = (probs > THRESHOLD).int()

        gathered_preds = accelerator.gather(preds).detach().cpu()
        gathered_labels = accelerator.gather(batch["labels"]).detach().cpu().int()

        all_preds.append(gathered_preds)
        all_labels.append(gathered_labels)

        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / len(dataloader)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    precision, recall, f1 = compute_micro_f1(all_preds, all_labels)
    return avg_loss, precision, recall, f1


# =========================
# 3. Main
# =========================
def main():
    print("===== Step 1: Initialize Accelerator =====")
    accelerator = Accelerator(mixed_precision="fp16")

    print("===== Step 2: Load Image Processor =====")
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    print("===== Step 3: Load Base Model =====")
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True
    )

    print("===== Step 4: Inject LoRA =====")
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("===== Step 5: Prepare Dataset =====")
    train_dataset = CocoPersonCarMultiLabelDataset(
        csv_file=TRAIN_CSV,
        image_root=IMAGE_ROOT,
        processor=image_processor,
        train=True
    )

    val_dataset = CocoPersonCarMultiLabelDataset(
        csv_file=VAL_CSV,
        image_root=IMAGE_ROOT,
        processor=image_processor,
        train=False
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    print("===== Step 6: Prepare DataLoader =====")
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

    print("===== Step 7: Loss / Optimizer / Scheduler =====")
    criterion = nn.BCEWithLogitsLoss()

    # Only optimize trainable LoRA parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    total_training_steps = EPOCHS * len(train_loader)
    warmup_steps = int(WARMUP_RATIO * total_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )

    print("===== Step 8: Accelerator Prepare =====")
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    best_val_f1 = -1.0
    log_file = os.path.join(LOG_OUTPUT_DIR, "train_log.txt")

    print("===== Step 9: Start Training =====")
    for epoch in range(EPOCHS):
        print(f"\n========== Epoch {epoch + 1}/{EPOCHS} ==========")

        train_loss, train_precision, train_recall, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, scheduler, accelerator
        )

        val_loss, val_precision, val_recall, val_f1 = validate_one_epoch(
            model, val_loader, criterion, accelerator
        )

        log_text = (
            f"Epoch {epoch + 1}/{EPOCHS}\n"
            f"Train Loss: {train_loss:.4f} | Train Precision: {train_precision:.4f} | "
            f"Train Recall: {train_recall:.4f} | Train F1: {train_f1:.4f}\n"
            f"Val   Loss: {val_loss:.4f} | Val   Precision: {val_precision:.4f} | "
            f"Val   Recall: {val_recall:.4f} | Val   F1: {val_f1:.4f}\n"
            + "-" * 80 + "\n"
        )

        if accelerator.is_main_process:
            print(log_text)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(log_text)

        # Save best PEFT model config + processor
        if accelerator.is_main_process and val_f1 > best_val_f1:
            best_val_f1 = val_f1

            unwrapped_model = accelerator.unwrap_model(model)

            # save previous peft model config and image processor
            unwrapped_model.save_pretrained(PEFT_MODEL_ID)
            image_processor.save_pretrained(PEFT_MODEL_ID)

            print(f"Best PEFT model saved to: {PEFT_MODEL_ID}")

    if accelerator.is_main_process:
        print(f"\nTraining finished. Best Val F1: {best_val_f1:.4f}")
        print(f"Saved model directory: {PEFT_MODEL_ID}")
        print(f"Log file: {log_file}")


if __name__ == "__main__":
    main()