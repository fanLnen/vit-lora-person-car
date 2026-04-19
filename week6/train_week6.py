import argparse
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from peft import get_peft_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification

from dataset import CocoPersonCarDataset
from utils import (
    DEFAULT_CLASS_NAMES,
    build_markdown_report,
    compute_multilabel_metrics,
    compute_singlelabel_metrics,
    count_parameters,
    get_lora_config,
    plot_confidence_distribution,
    plot_curve,
    plot_error_type_distribution,
    plot_loss_accuracy_relation,
    plot_multilabel_confusion_matrices,
    plot_single_confusion_matrix,
    save_json,
    save_markdown,
    timestamp_string,
    try_get_gpu_utilization,
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


def parse_args():
    parser = argparse.ArgumentParser(description="Week 6: full training, evaluation, checkpointing, and error analysis")

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

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3, help="early stopping patience based on validation accuracy")
    parser.add_argument("--difficult_threshold", type=float, default=0.15, help="distance-to-threshold for difficult samples")

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--best_config_json",
        type=str,
        default="",
        help="Optional path to Week 5 best_hyperparameters.json. If provided, lora_r/lora_alpha/lr are loaded automatically.",
    )

    parser.add_argument("--use_scheduler", action="store_true")
    parser.add_argument("--experiment_id", type=str, default="")
    parser.add_argument("--class_names", nargs="+", default=DEFAULT_CLASS_NAMES)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--output_dir", type=str, default=os.path.join(base_dir, "outputs_week6"))

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_problem_type(mode: str):
    return "multi_label_classification" if mode == "multilabel" else "single_label_classification"


def get_criterion(mode: str):
    return nn.BCEWithLogitsLoss() if mode == "multilabel" else nn.CrossEntropyLoss()


def maybe_load_best_config(args):
    if args.best_config_json and os.path.exists(args.best_config_json):
        import json

        with open(args.best_config_json, "r", encoding="utf-8") as f:
            best_cfg = json.load(f)
        args.lora_r = int(best_cfg["lora_r"])
        args.lora_alpha = int(best_cfg["lora_alpha"])
        args.lr = float(best_cfg["lr"])


def build_model(args):
    base_model = AutoModelForImageClassification.from_pretrained(
        args.model_name,
        num_labels=args.num_labels,
        problem_type=get_problem_type(args.mode),
        ignore_mismatched_sizes=True,
    )
    lora_config = get_lora_config(lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    model = get_peft_model(base_model, lora_config)
    return model


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
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def train_one_epoch(model, dataloader, optimizer, criterion, device, mode):
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

    return running_loss / max(1, len(dataloader))


@torch.no_grad()
def collect_predictions(model, dataloader, criterion, device, mode, threshold):
    model.eval()

    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    all_paths = []

    for batch in tqdm(dataloader, desc="Validation", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        file_paths = batch["file_path"]

        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits

        if mode == "multilabel":
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).int()
        else:
            loss = criterion(logits, labels.long())
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

        running_loss += loss.item()
        all_labels.append(labels.detach().cpu())
        all_preds.append(preds.detach().cpu())
        all_probs.append(probs.detach().cpu())
        all_paths.extend(list(file_paths))

    avg_loss = running_loss / max(1, len(dataloader))
    labels_tensor = torch.cat(all_labels, dim=0)
    preds_tensor = torch.cat(all_preds, dim=0)
    probs_tensor = torch.cat(all_probs, dim=0)

    return avg_loss, labels_tensor, preds_tensor, probs_tensor, all_paths


def evaluate_predictions(mode, labels_tensor, preds_tensor, probs_tensor, class_names):
    y_true = labels_tensor.numpy()
    y_pred = preds_tensor.numpy()
    y_prob = probs_tensor.numpy()

    if mode == "multilabel":
        return compute_multilabel_metrics(y_true, y_pred, y_prob, class_names=class_names)
    return compute_singlelabel_metrics(y_true, y_pred, y_prob, class_names=class_names)


def save_epoch_checkpoint(
    checkpoint_path: str,
    epoch: int,
    model,
    optimizer,
    scheduler,
    training_state: Dict,
    args,
):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "training_state": training_state,
            "args": vars(args),
        },
        checkpoint_path,
    )


def restore_checkpoint(model, optimizer, scheduler, checkpoint_path: str, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def save_best_adapter_bundle(model, processor, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)


def perform_error_analysis(
    args,
    labels_tensor,
    preds_tensor,
    probs_tensor,
    file_paths: List[str],
    output_dir: str,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    y_true = labels_tensor.numpy()
    y_pred = preds_tensor.numpy()
    y_prob = probs_tensor.numpy()

    rows = []
    error_counts = {
        "correct": 0,
        "false_positive": 0,
        "false_negative": 0,
        "wrong_class": 0,
        "difficult_sample": 0,
        "mixed_error": 0,
    }
    conf_correct = []
    conf_incorrect = []

    if args.mode == "multilabel":
        for i, path in enumerate(file_paths):
            gt = y_true[i].astype(int).tolist()
            pred = y_pred[i].astype(int).tolist()
            prob = y_prob[i].tolist()
            per_label_types = []

            for c in range(args.num_labels):
                if gt[c] == 0 and pred[c] == 1:
                    per_label_types.append("false_positive")
                elif gt[c] == 1 and pred[c] == 0:
                    per_label_types.append("false_negative")
                else:
                    per_label_types.append("correct")

            is_correct = gt == pred
            margins = [abs(p - args.threshold) for p in prob]
            low_conf = min(margins) < args.difficult_threshold

            if is_correct:
                primary_error = "correct"
                error_counts["correct"] += 1
                conf_correct.append(float(max(prob)))
            else:
                unique_errors = {x for x in per_label_types if x != "correct"}
                if low_conf:
                    primary_error = "difficult_sample"
                    error_counts["difficult_sample"] += 1
                elif len(unique_errors) == 1:
                    primary_error = list(unique_errors)[0]
                    error_counts[primary_error] += 1
                else:
                    primary_error = "mixed_error"
                    error_counts["mixed_error"] += 1
                conf_incorrect.append(float(max(prob)))

            row = {
                "file_path": path,
                "is_correct": is_correct,
                "primary_error_type": primary_error,
                "sample_confidence": float(max(prob)),
            }
            for c, class_name in enumerate(args.class_names[: args.num_labels]):
                row[f"{class_name}_gt"] = gt[c]
                row[f"{class_name}_pred"] = pred[c]
                row[f"{class_name}_confidence"] = float(prob[c])
                row[f"{class_name}_error_type"] = per_label_types[c]
            rows.append(row)

    else:
        for i, path in enumerate(file_paths):
            gt = int(y_true[i])
            pred = int(y_pred[i])
            prob = y_prob[i].tolist()
            best_conf = float(max(prob))
            low_conf = abs(best_conf - 0.5) < args.difficult_threshold
            is_correct = gt == pred

            if is_correct:
                primary_error = "correct"
                error_counts["correct"] += 1
                conf_correct.append(best_conf)
            else:
                if low_conf:
                    primary_error = "difficult_sample"
                    error_counts["difficult_sample"] += 1
                else:
                    primary_error = "wrong_class"
                    error_counts["wrong_class"] += 1
                conf_incorrect.append(best_conf)

            row = {
                "file_path": path,
                "true_label": gt,
                "pred_label": pred,
                "is_correct": is_correct,
                "primary_error_type": primary_error,
                "sample_confidence": best_conf,
            }
            for c, class_name in enumerate(args.class_names[: args.num_labels]):
                row[f"prob_{class_name}"] = float(prob[c])
            rows.append(row)

    error_df = pd.DataFrame(rows)
    error_csv = os.path.join(output_dir, "error_analysis_details.csv")
    error_df.to_csv(error_csv, index=False)

    plot_error_type_distribution(error_counts, os.path.join(output_dir, "error_type_distribution.png"))
    plot_confidence_distribution(
        conf_correct,
        conf_incorrect,
        os.path.join(output_dir, "confidence_distribution.png"),
    )

    summary = {
        "total_samples": int(len(file_paths)),
        "correct_samples": int(error_counts["correct"]),
        "incorrect_samples": int(len(file_paths) - error_counts["correct"]),
        "error_type_counts": error_counts,
        "error_details_csv": error_csv,
    }
    save_json(summary, os.path.join(output_dir, "error_analysis_summary.json"))
    return summary


def create_training_visualizations(training_state: Dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    epochs = list(range(1, len(training_state["train_losses"]) + 1))
    best_epoch = int(training_state["epoch_of_best_accuracy"])

    plot_curve(
        epochs,
        training_state["train_losses"],
        "Epoch",
        "Loss",
        "Training Loss",
        os.path.join(output_dir, "train_loss_curve.png"),
        best_x=best_epoch if 1 <= best_epoch <= len(epochs) else None,
        best_y=training_state["train_losses"][best_epoch - 1] if 1 <= best_epoch <= len(epochs) else None,
    )
    plot_curve(
        epochs,
        training_state["val_losses"],
        "Epoch",
        "Loss",
        "Validation Loss",
        os.path.join(output_dir, "val_loss_curve.png"),
        best_x=best_epoch if 1 <= best_epoch <= len(epochs) else None,
        best_y=training_state["val_losses"][best_epoch - 1] if 1 <= best_epoch <= len(epochs) else None,
    )
    plot_curve(
        epochs,
        training_state["val_accuracies"],
        "Epoch",
        "Accuracy",
        "Validation Accuracy",
        os.path.join(output_dir, "val_accuracy_curve.png"),
        best_x=best_epoch if 1 <= best_epoch <= len(epochs) else None,
        best_y=training_state["val_accuracies"][best_epoch - 1] if 1 <= best_epoch <= len(epochs) else None,
    )
    plot_curve(
        epochs,
        training_state["learning_rates"],
        "Epoch",
        "Learning Rate",
        "Learning Rate Schedule",
        os.path.join(output_dir, "learning_rate_curve.png"),
        best_x=best_epoch if 1 <= best_epoch <= len(epochs) else None,
        best_y=training_state["learning_rates"][best_epoch - 1] if 1 <= best_epoch <= len(epochs) else None,
    )
    plot_loss_accuracy_relation(
        training_state["val_losses"],
        training_state["val_accuracies"],
        best_epoch=best_epoch,
        save_path=os.path.join(output_dir, "loss_accuracy_relation.png"),
    )

    history_df = pd.DataFrame(
        {
            "epoch": epochs,
            "train_loss": training_state["train_losses"],
            "val_loss": training_state["val_losses"],
            "val_accuracy": training_state["val_accuracies"],
            "learning_rate": training_state["learning_rates"],
        }
    )
    history_df.to_csv(os.path.join(output_dir, "training_history.csv"), index=False)


def save_confusion_matrix_plots(args, metrics: Dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    if args.mode == "multilabel":
        plot_multilabel_confusion_matrices(metrics["confusion_matrix"], output_dir)
    else:
        plot_single_confusion_matrix(
            metrics["confusion_matrix"],
            args.class_names[: args.num_labels],
            "Confusion Matrix",
            os.path.join(output_dir, "confusion_matrix.png"),
        )


def main():
    args = parse_args()
    maybe_load_best_config(args)

    set_seed(args.seed)
    device = get_device()

    experiment_id = args.experiment_id or f"week6_{args.mode}_{timestamp_string()}"
    experiment_dir = os.path.join(args.output_dir, experiment_id)
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    plot_dir = os.path.join(experiment_dir, "plots")
    eval_dir = os.path.join(experiment_dir, "evaluation")
    error_dir = os.path.join(experiment_dir, "error_analysis")
    final_model_dir = os.path.join(experiment_dir, "best_model_adapter")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Experiment directory: {experiment_dir}")
    print(f"Mode: {args.mode}")
    print(f"LoRA config: r={args.lora_r}, alpha={args.lora_alpha}, lr={args.lr}")

    processor = AutoImageProcessor.from_pretrained(args.model_name)
    train_dataset, val_dataset = create_datasets(args, processor)
    train_loader = create_loader(train_dataset, args.batch_size, True, args.num_workers)
    val_loader = create_loader(val_dataset, args.batch_size, False, args.num_workers)

    model = build_model(args).to(device)
    criterion = get_criterion(args.mode)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1) if args.use_scheduler else None

    trainable_count, total_count, trainable_pct = count_parameters(model)

    training_state = {
        "epoch": 0,
        "best_val_loss": float("inf"),
        "best_val_accuracy": 0.0,
        "patience_counter": 0,
        "train_losses": [],
        "val_losses": [],
        "val_accuracies": [],
        "learning_rates": [],
        "gpu_utilizations": [],
        "epoch_of_best_accuracy": 0,
        "epoch_of_best_loss": 0,
    }

    start_time = time.time()

    best_accuracy_ckpt = os.path.join(checkpoint_dir, "best_val_accuracy.pt")
    best_loss_ckpt = os.path.join(checkpoint_dir, "best_val_loss.pt")

    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        gpu_util = try_get_gpu_utilization()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, args.mode)
        val_loss, labels_tensor, preds_tensor, probs_tensor, file_paths = collect_predictions(
            model, val_loader, criterion, device, args.mode, args.threshold
        )
        val_metrics = evaluate_predictions(
            args.mode,
            labels_tensor,
            preds_tensor,
            probs_tensor,
            class_names=args.class_names[: args.num_labels],
        )
        val_accuracy = float(val_metrics["accuracy"])

        if scheduler is not None:
            scheduler.step(val_loss)

        training_state["epoch"] = epoch
        training_state["train_losses"].append(float(train_loss))
        training_state["val_losses"].append(float(val_loss))
        training_state["val_accuracies"].append(val_accuracy)
        training_state["learning_rates"].append(float(current_lr))
        training_state["gpu_utilizations"].append(gpu_util)

        epoch_ckpt = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pt")
        save_epoch_checkpoint(epoch_ckpt, epoch, model, optimizer, scheduler, training_state, args)

        improved_accuracy = val_accuracy > training_state["best_val_accuracy"]
        improved_loss = val_loss < training_state["best_val_loss"]

        if improved_accuracy:
            training_state["best_val_accuracy"] = val_accuracy
            training_state["patience_counter"] = 0
            training_state["epoch_of_best_accuracy"] = epoch
            save_epoch_checkpoint(best_accuracy_ckpt, epoch, model, optimizer, scheduler, training_state, args)
            save_best_adapter_bundle(model, processor, final_model_dir)
        else:
            training_state["patience_counter"] += 1

        if improved_loss:
            training_state["best_val_loss"] = float(val_loss)
            training_state["epoch_of_best_loss"] = epoch
            save_epoch_checkpoint(best_loss_ckpt, epoch, model, optimizer, scheduler, training_state, args)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | val_acc={val_accuracy:.6f} | "
            f"best_val_acc={training_state['best_val_accuracy']:.6f} | patience={training_state['patience_counter']}/{args.patience}"
        )

        if training_state["patience_counter"] >= args.patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    training_time_hours = (time.time() - start_time) / 3600.0

    restore_checkpoint(model, None, None, best_accuracy_ckpt, device)

    final_val_loss, labels_tensor, preds_tensor, probs_tensor, file_paths = collect_predictions(
        model, val_loader, criterion, device, args.mode, args.threshold
    )
    final_metrics = evaluate_predictions(
        args.mode,
        labels_tensor,
        preds_tensor,
        probs_tensor,
        class_names=args.class_names[: args.num_labels],
    )

    final_metrics["final_val_loss"] = float(final_val_loss)
    save_json(final_metrics, os.path.join(eval_dir, "validation_metrics.json"))
    save_confusion_matrix_plots(args, final_metrics, os.path.join(eval_dir, "confusion_matrices"))

    error_summary = perform_error_analysis(args, labels_tensor, preds_tensor, probs_tensor, file_paths, error_dir)
    create_training_visualizations(training_state, plot_dir)

    peak_gpu_memory_gb = None
    if torch.cuda.is_available():
        peak_gpu_memory_gb = float(torch.cuda.max_memory_allocated() / (1024 ** 3))

    valid_gpu_utils = [x for x in training_state["gpu_utilizations"] if x is not None]
    avg_gpu_util = float(sum(valid_gpu_utils) / len(valid_gpu_utils)) if valid_gpu_utils else None

    report = {
        "experiment_summary": {
            "experiment_id": experiment_id,
            "completion_date": timestamp_string(),
            "total_training_epochs": int(training_state["epoch"]),
            "best_epoch": int(training_state["epoch_of_best_accuracy"]),
            "training_time_hours": round(training_time_hours, 4),
        },
        "hyperparameters": {
            "model_name": args.model_name,
            "mode": args.mode,
            "num_labels": args.num_labels,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "threshold": args.threshold,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "use_scheduler": args.use_scheduler,
        },
        "performance_metrics": final_metrics,
        "training_metrics": {
            "final_train_loss": float(training_state["train_losses"][-1]),
            "final_val_loss": float(final_val_loss),
            "best_val_accuracy": float(training_state["best_val_accuracy"]),
            "best_val_loss": float(training_state["best_val_loss"]),
        },
        "error_analysis_summary": error_summary,
        "model_information": {
            "model_type": "ViT + LoRA",
            "trainable_parameters": int(trainable_count),
            "total_parameters": int(total_count),
            "trainable_percentage": round(trainable_pct, 4),
        },
        "hardware_utilization": {
            "peak_gpu_memory_gb": peak_gpu_memory_gb,
            "average_gpu_utilization": avg_gpu_util,
        },
    }

    report_json_path = os.path.join(experiment_dir, "final_evaluation_report.json")
    report_md_path = os.path.join(experiment_dir, "final_evaluation_report.md")
    save_json(report, report_json_path)
    save_markdown(build_markdown_report(report), report_md_path)
    save_json(training_state, os.path.join(experiment_dir, "training_state.json"))

    print("\nWeek 6 finished.")
    print(f"Best epoch: {training_state['epoch_of_best_accuracy']}")
    print(f"Best validation accuracy: {training_state['best_val_accuracy']:.6f}")
    print(f"Report saved to: {report_json_path}")
    print(f"Markdown report saved to: {report_md_path}")
    print(f"Best adapter saved to: {final_model_dir}")


if __name__ == "__main__":
    main()
