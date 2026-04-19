import argparse
import itertools
import math
import os
import random
from copy import deepcopy

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from peft import get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification

from dataset import CocoPersonCarDataset
from utils import (
    compute_multilabel_metrics,
    compute_singlelabel_metrics,
    get_lora_config,
    save_json,
)


# =========================
# 1. Default Config
# =========================
TRAIN_CSV = r"D:\Praktikum_DJi\output\coco_person_car_train.csv"
VAL_CSV = r"D:\Praktikum_DJi\output\coco_person_car_val.csv"
IMAGE_ROOT = r"D:\Praktikum_DJi\coco"

MODEL_NAME = "google/vit-base-patch16-224"
NUM_LABELS = 2
BATCH_SIZE = 8
NUM_WORKERS = 0
WEIGHT_DECAY = 1e-2
THRESHOLD = 0.5
SEED = 42

PARAM_COMBINATIONS = [
    {"lora_r": 8, "lora_alpha": 16, "lr": 1e-4},
    {"lora_r": 16, "lora_alpha": 32, "lr": 3e-4},
    {"lora_r": 32, "lora_alpha": 64, "lr": 1e-3},
    {"lora_r": 64, "lora_alpha": 128, "lr": 3e-3},
]


# =========================
# 2. Helpers
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Week 5: initial training and hyperparameter tuning")

    parser.add_argument("--mode", type=str, default="multilabel", choices=["multilabel", "singlelabel"])
    parser.add_argument("--train_csv", type=str, default=TRAIN_CSV)
    parser.add_argument("--val_csv", type=str, default=VAL_CSV)
    parser.add_argument("--image_root", type=str, default=IMAGE_ROOT)
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--num_labels", type=int, default=NUM_LABELS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--seed", type=int, default=SEED)

    parser.add_argument("--quick_subset_ratio", type=float, default=0.02, help="1%-5% is recommended")
    parser.add_argument("--quick_epochs", type=int, default=2)

    parser.add_argument("--lr_min", type=float, default=1e-7)
    parser.add_argument("--lr_max", type=float, default=10.0)
    parser.add_argument("--num_iter", type=int, default=100)

    parser.add_argument("--sweep_subset_ratio", type=float, default=0.05, help="screening subset ratio")
    parser.add_argument("--sweep_epochs", type=int, default=2)

    parser.add_argument("--run_quick_test", action="store_true")
    parser.add_argument("--run_lr_finder", action="store_true")
    parser.add_argument("--run_sweep", action="store_true")
    parser.add_argument("--run_all", action="store_true")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--output_dir", type=str, default=os.path.join(base_dir, "outputs_week5"))

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_problem_type(mode: str):
    return "multi_label_classification" if mode == "multilabel" else "single_label_classification"


def get_criterion(mode: str):
    if mode == "multilabel":
        return nn.BCEWithLogitsLoss()
    return nn.CrossEntropyLoss()


def build_model(args, lora_r: int, lora_alpha: int):
    base_model = AutoModelForImageClassification.from_pretrained(
        args.model_name,
        num_labels=args.num_labels,
        problem_type=get_problem_type(args.mode),
        ignore_mismatched_sizes=True,
    )
    lora_config = get_lora_config(lora_r=lora_r, lora_alpha=lora_alpha)
    model = get_peft_model(base_model, lora_config)
    return model


def make_subset(dataset, ratio: float, seed: int):
    ratio = max(0.0, min(1.0, ratio))
    subset_size = max(1, int(len(dataset) * ratio))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
    return Subset(dataset, indices)


def create_datasets(args, processor):
    train_dataset = CocoPersonCarDataset(
        csv_file=args.train_csv,
        image_root=args.image_root,
        processor=processor,
        mode=args.mode,
        train=True,
    )
    val_dataset = CocoPersonCarDataset(
        csv_file=args.val_csv,
        image_root=args.image_root,
        processor=processor,
        mode=args.mode,
        train=False,
    )
    return train_dataset, val_dataset


def create_loader(dataset, batch_size: int, shuffle: bool, num_workers: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def evaluate_model(model, dataloader, criterion, device, mode, threshold, num_labels):
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values)

            if mode == "multilabel":
                loss = criterion(outputs.logits, labels)
                probs = torch.sigmoid(outputs.logits)
                preds = (probs > threshold).int()
            else:
                loss = criterion(outputs.logits, labels.long())
                preds = torch.argmax(outputs.logits, dim=1)

            running_loss += loss.item()
            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())

    avg_loss = running_loss / max(1, len(dataloader))
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    if mode == "multilabel":
        accuracy, precision, recall, f1 = compute_multilabel_metrics(all_preds, all_labels)
        return {
            "val_loss": avg_loss,
            "val_accuracy": accuracy,
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": f1,
        }

    accuracy, precision, recall, f1, per_class_metrics = compute_singlelabel_metrics(
        all_preds, all_labels, num_classes=num_labels
    )
    return {
        "val_loss": avg_loss,
        "val_accuracy": accuracy,
        "val_precision": precision,
        "val_recall": recall,
        "val_f1": f1,
        "per_class_metrics": per_class_metrics,
    }


def train_one_epoch(model, dataloader, optimizer, criterion, device, mode, threshold):
    model.train()
    running_loss = 0.0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        optimizer.zero_grad()

        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(pixel_values=pixel_values)

        if mode == "multilabel":
            loss = criterion(outputs.logits, labels)
        else:
            loss = criterion(outputs.logits, labels.long())

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / max(1, len(dataloader))
    return avg_loss


def run_quick_test(args, device):
    print("\n===== 1) Small-subset quick test =====")
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    train_dataset, _ = create_datasets(args, processor)

    quick_subset = make_subset(train_dataset, args.quick_subset_ratio, args.seed)
    quick_loader = create_loader(
        quick_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    model = build_model(args, lora_r=8, lora_alpha=16).to(device)
    criterion = get_criterion(args.mode)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=1e-4, weight_decay=args.weight_decay)

    records = []
    for epoch in range(1, args.quick_epochs + 1):
        train_loss = train_one_epoch(
            model, quick_loader, optimizer, criterion, device, args.mode, args.threshold
        )
        print(f"Quick Test Epoch {epoch}/{args.quick_epochs} - train_loss: {train_loss:.6f}")
        records.append({"epoch": epoch, "train_loss": train_loss, "subset_size": len(quick_subset)})

    quick_test_csv = os.path.join(args.output_dir, "quick_test_results.csv")
    pd.DataFrame(records).to_csv(quick_test_csv, index=False)

    quick_test_summary = {
        "subset_ratio": args.quick_subset_ratio,
        "subset_size": len(quick_subset),
        "epochs": args.quick_epochs,
        "initial_train_loss": records[0]["train_loss"],
        "final_train_loss": records[-1]["train_loss"],
        "loss_decreased": records[-1]["train_loss"] < records[0]["train_loss"],
    }
    save_json(quick_test_summary, os.path.join(args.output_dir, "quick_test_summary.json"))

    return quick_test_summary


def run_lr_finder(args, device):
    print("\n===== 2) Learning rate finder =====")
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    train_dataset, _ = create_datasets(args, processor)

    lr_subset = make_subset(train_dataset, args.quick_subset_ratio, args.seed)
    lr_loader = create_loader(
        lr_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    model = build_model(args, lora_r=8, lora_alpha=16).to(device)
    criterion = get_criterion(args.mode)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr_min, weight_decay=args.weight_decay)

    lr_mult = (args.lr_max / args.lr_min) ** (1 / max(1, args.num_iter - 1))
    iterator = itertools.cycle(lr_loader)

    best_smoothed_loss = float("inf")
    avg_loss = 0.0
    beta = 0.98

    history = []

    model.train()
    for iteration in range(1, args.num_iter + 1):
        batch = next(iterator)

        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values)

        if args.mode == "multilabel":
            loss = criterion(outputs.logits, labels)
        else:
            loss = criterion(outputs.logits, labels.long())

        loss.backward()
        optimizer.step()

        current_lr = optimizer.param_groups[0]["lr"]

        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** iteration)

        if smoothed_loss < best_smoothed_loss:
            best_smoothed_loss = smoothed_loss

        history.append(
            {
                "iteration": iteration,
                "lr": current_lr,
                "loss": loss.item(),
                "smoothed_loss": smoothed_loss,
            }
        )

        optimizer.param_groups[0]["lr"] *= lr_mult

    lr_df = pd.DataFrame(history)
    lr_csv = os.path.join(args.output_dir, "lr_finder_results.csv")
    lr_df.to_csv(lr_csv, index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(lr_df["lr"], lr_df["smoothed_loss"])
    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Smoothed Loss")
    plt.title("Learning Rate Finder")
    plt.grid(True)
    lr_plot = os.path.join(args.output_dir, "lr_finder_curve.png")
    plt.tight_layout()
    plt.savefig(lr_plot, dpi=200)
    plt.close()

    min_idx = lr_df["smoothed_loss"].idxmin()
    best_lr_at_min_loss = float(lr_df.loc[min_idx, "lr"])

    recommended_low = best_lr_at_min_loss / 10
    recommended_high = best_lr_at_min_loss / 3

    summary = {
        "lr_min": args.lr_min,
        "lr_max": args.lr_max,
        "num_iter": args.num_iter,
        "best_lr_at_min_smoothed_loss": best_lr_at_min_loss,
        "recommended_lr_low": recommended_low,
        "recommended_lr_high": recommended_high,
    }
    save_json(summary, os.path.join(args.output_dir, "lr_finder_summary.json"))

    print(
        f"LR Finder done. Recommended initial LR range: "
        f"{recommended_low:.2e} ~ {recommended_high:.2e}"
    )
    return summary


def train_and_validate_for_params(args, device, params, train_dataset, val_dataset):
    train_subset = make_subset(train_dataset, args.sweep_subset_ratio, args.seed)
    train_loader = create_loader(
        train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = create_loader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = build_model(args, lora_r=params["lora_r"], lora_alpha=params["lora_alpha"]).to(device)

    criterion = get_criterion(args.mode)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=params["lr"], weight_decay=args.weight_decay)

    epoch_records = []
    for epoch in range(1, args.sweep_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args.mode, args.threshold
        )
        val_metrics = evaluate_model(
            model, val_loader, criterion, device, args.mode, args.threshold, args.num_labels
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
        }
        epoch_records.append(record)

        print(
            f"Params {params} | Epoch {epoch}/{args.sweep_epochs} | "
            f"train_loss={train_loss:.6f} | val_loss={val_metrics['val_loss']:.6f} | "
            f"val_acc={val_metrics['val_accuracy']:.6f}"
        )

    final_record = deepcopy(epoch_records[-1])
    final_record.update(params)
    final_record["train_subset_size"] = len(train_subset)
    return final_record, epoch_records


def run_hyperparameter_sweep(args, device):
    print("\n===== 3) Hyperparameter sweep =====")
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    train_dataset, val_dataset = create_datasets(args, processor)

    all_final_results = []
    all_epoch_rows = []

    for params in PARAM_COMBINATIONS:
        final_result, epoch_records = train_and_validate_for_params(
            args, device, params, train_dataset, val_dataset
        )
        all_final_results.append(final_result)

        for row in epoch_records:
            row_copy = dict(row)
            row_copy.update(params)
            all_epoch_rows.append(row_copy)

    final_df = pd.DataFrame(all_final_results)
    epoch_df = pd.DataFrame(all_epoch_rows)

    final_csv = os.path.join(args.output_dir, "hyperparameter_results.csv")
    epoch_csv = os.path.join(args.output_dir, "hyperparameter_epoch_results.csv")
    final_df.to_csv(final_csv, index=False)
    epoch_df.to_csv(epoch_csv, index=False)

    sorted_df = final_df.sort_values(by=["val_accuracy", "val_loss"], ascending=[False, True]).reset_index(drop=True)
    best_row = sorted_df.iloc[0].to_dict()

    best_config = {
        "lora_r": int(best_row["lora_r"]),
        "lora_alpha": int(best_row["lora_alpha"]),
        "lr": float(best_row["lr"]),
        "val_accuracy": float(best_row["val_accuracy"]),
        "val_loss": float(best_row["val_loss"]),
        "val_f1": float(best_row["val_f1"]),
    }
    save_json(best_config, os.path.join(args.output_dir, "best_hyperparameters.json"))

    print(f"Best config: {best_config}")
    return best_config


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    device = get_device()

    print(f"Device: {device}")
    print(f"Output directory: {args.output_dir}")
    print(f"Mode: {args.mode}")

    if not (args.run_quick_test or args.run_lr_finder or args.run_sweep or args.run_all):
        raise ValueError(
            "Please choose at least one run mode: --run_quick_test, --run_lr_finder, --run_sweep, or --run_all"
        )

    if args.run_all or args.run_quick_test:
        run_quick_test(args, device)

    if args.run_all or args.run_lr_finder:
        run_lr_finder(args, device)

    if args.run_all or args.run_sweep:
        run_hyperparameter_sweep(args, device)

    print("\nWeek 5 experiments finished.")


if __name__ == "__main__":
    main()
