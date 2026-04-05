import os
import argparse
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

from dataset import CocoPersonCarDataset
from utils import (
    compute_multilabel_metrics,
    compute_singlelabel_metrics,
    get_lora_config
)


# =========================
# 1. Config
# =========================
TRAIN_CSV = r"D:\Praktikum_DJi\output\coco_person_car_train.csv"
VAL_CSV = r"D:\Praktikum_DJi\output\coco_person_car_val.csv"
IMAGE_ROOT = r"D:\Praktikum_DJi\coco"

MODEL_NAME = "google/vit-base-patch16-224"
NUM_LABELS = 2

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
# 2. Argument Parser
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Train ViT + LoRA for person/car classification")
    parser.add_argument(
        "--mode",
        type=str,
        default="multilabel",
        choices=["multilabel", "singlelabel"],
        help="Training mode: multilabel or singlelabel"
    )
    return parser.parse_args()


# =========================
# 3. Train / Validation
# =========================
def train_one_epoch(model, dataloader, optimizer, criterion, scheduler, accelerator, mode):
    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        optimizer.zero_grad()

        pixel_values = batch["pixel_values"]
        labels = batch["labels"]

        outputs = model(pixel_values=pixel_values)

        if mode == "multilabel":
            loss = criterion(outputs.logits, labels)
            probs = torch.sigmoid(outputs.logits)
            preds = (probs > THRESHOLD).int()
        else:
            loss = criterion(outputs.logits, labels.long())
            preds = torch.argmax(outputs.logits, dim=1)

        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        gathered_preds = accelerator.gather(preds).detach().cpu()
        gathered_labels = accelerator.gather(labels).detach().cpu()

        all_preds.append(gathered_preds)
        all_labels.append(gathered_labels)

        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / len(dataloader)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    if mode == "multilabel":
        accuracy, precision, recall, f1 = compute_multilabel_metrics(all_preds, all_labels)
        metrics_text = (
            f"Train Loss: {avg_loss:.4f} | "
            f"Train Acc: {accuracy:.4f} | Train Precision: {precision:.4f} | "
            f"Train Recall: {recall:.4f} | Train F1: {f1:.4f}"
        )
        return avg_loss, accuracy, precision, recall, f1, metrics_text
    else:
        accuracy, precision, recall, f1, per_class_metrics = compute_singlelabel_metrics(
            all_preds, all_labels, num_classes=NUM_LABELS
        )
        metrics_text = (
            f"Train Loss: {avg_loss:.4f} | "
            f"Train Acc: {accuracy:.4f} | Train Macro Precision: {precision:.4f} | "
            f"Train Macro Recall: {recall:.4f} | Train Macro F1: {f1:.4f}\n"
            f"Train Per-class metrics: {per_class_metrics}"
        )
        return avg_loss, accuracy, precision, recall, f1, metrics_text


@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, accelerator, mode):
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Validation", leave=False)

    for batch in progress_bar:
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]

        outputs = model(pixel_values=pixel_values)

        if mode == "multilabel":
            loss = criterion(outputs.logits, labels)
            probs = torch.sigmoid(outputs.logits)
            preds = (probs > THRESHOLD).int()
        else:
            loss = criterion(outputs.logits, labels.long())
            preds = torch.argmax(outputs.logits, dim=1)

        running_loss += loss.item()

        gathered_preds = accelerator.gather(preds).detach().cpu()
        gathered_labels = accelerator.gather(labels).detach().cpu()

        all_preds.append(gathered_preds)
        all_labels.append(gathered_labels)

        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / len(dataloader)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    if mode == "multilabel":
        accuracy, precision, recall, f1 = compute_multilabel_metrics(all_preds, all_labels)
        metrics_text = (
            f"Val   Loss: {avg_loss:.4f} | "
            f"Val   Acc: {accuracy:.4f} | Val   Precision: {precision:.4f} | "
            f"Val   Recall: {recall:.4f} | Val   F1: {f1:.4f}"
        )
        return avg_loss, accuracy, precision, recall, f1, metrics_text
    else:
        accuracy, precision, recall, f1, per_class_metrics = compute_singlelabel_metrics(
            all_preds, all_labels, num_classes=NUM_LABELS
        )
        metrics_text = (
            f"Val   Loss: {avg_loss:.4f} | "
            f"Val   Acc: {accuracy:.4f} | Val   Macro Precision: {precision:.4f} | "
            f"Val   Macro Recall: {recall:.4f} | Val   Macro F1: {f1:.4f}\n"
            f"Val   Per-class metrics: {per_class_metrics}"
        )
        return avg_loss, accuracy, precision, recall, f1, metrics_text


# =========================
# 4. Main
# =========================
def main():
    args = parse_args()

    print("===== Step 1: Initialize Accelerator =====")
    accelerator = Accelerator(mixed_precision="fp16")

    print("===== Step 2: Load Image Processor =====")
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    print("===== Step 3: Load Base Model =====")
    problem_type = "multi_label_classification" if args.mode == "multilabel" else "single_label_classification"
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        problem_type=problem_type,
        ignore_mismatched_sizes=True
    )

    print("===== Step 4: Inject LoRA =====")
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("===== Step 5: Prepare Dataset =====")
    train_dataset = CocoPersonCarDataset(
        csv_file=TRAIN_CSV,
        image_root=IMAGE_ROOT,
        processor=image_processor,
        mode=args.mode,
        train=True
    )

    val_dataset = CocoPersonCarDataset(
        csv_file=VAL_CSV,
        image_root=IMAGE_ROOT,
        processor=image_processor,
        mode=args.mode,
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
    if args.mode == "multilabel":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

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
    log_file = os.path.join(LOG_OUTPUT_DIR, f"train_log_{args.mode}.txt")

    print("===== Step 9: Start Training =====")
    for epoch in range(EPOCHS):
        print(f"\n========== Epoch {epoch + 1}/{EPOCHS} | Mode: {args.mode} ==========")

        train_loss, train_acc, train_precision, train_recall, train_f1, train_metrics_text = train_one_epoch(
            model, train_loader, optimizer, criterion, scheduler, accelerator, args.mode
        )

        val_loss, val_acc, val_precision, val_recall, val_f1, val_metrics_text = validate_one_epoch(
            model, val_loader, criterion, accelerator, args.mode
        )

        log_text = (
            f"Epoch {epoch + 1}/{EPOCHS} | Mode: {args.mode}\n"
            f"{train_metrics_text}\n"
            f"{val_metrics_text}\n"
            + "-" * 100 + "\n"
        )

        if accelerator.is_main_process:
            print(log_text)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(log_text)

        # Save best PEFT model config + processor
        if accelerator.is_main_process and val_f1 > best_val_f1:
            best_val_f1 = val_f1

            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(PEFT_MODEL_ID)
            image_processor.save_pretrained(PEFT_MODEL_ID)

            print(f"Best PEFT model saved to: {PEFT_MODEL_ID}")

    if accelerator.is_main_process:
        print(f"\nTraining finished. Best Val F1: {best_val_f1:.4f}")
        print(f"Saved model directory: {PEFT_MODEL_ID}")
        print(f"Log file: {log_file}")


if __name__ == "__main__":
    main()