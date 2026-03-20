import os
import ast
import random
import torch
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from peft import LoraConfig, get_peft_model



# 1. 配置

VAL_CSV = r"D:\Praktikum_DJi\output\coco_person_car_val.csv"
IMAGE_ROOT = r"D:\Praktikum_DJi\coco"
MODEL_NAME = "google/vit-base-patch16-224"
CHECKPOINT_PATH = r"D:\Praktikum_DJi\output\checkpoints\best_vit_lora_person_car.pt"

THRESHOLD = 0.5
CLASS_NAMES = ["person", "car"]
NUM_SAMPLES = 9   # 显示 9 张图



# 2. 构建模型

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



# 3. 标签转文字

def label_to_text(label_list):
    names = []
    for i, v in enumerate(label_list):
        if v == 1:
            names.append(CLASS_NAMES[i])
    if len(names) == 0:
        return "none"
    return ",".join(names)



# 4. 主函数

def main():
    df = pd.read_csv(VAL_CSV)
    samples = df.sample(n=min(NUM_SAMPLES, len(df)), random_state=42)

    processor, model = build_model()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    n = len(samples)
    cols = 3
    rows = (n + cols - 1) // cols

    plt.figure(figsize=(15, 5 * rows))

    for idx, (_, row) in enumerate(samples.iterrows(), start=1):
        relative_path = row["file_path"]
        image_path = os.path.join(IMAGE_ROOT, relative_path)

        true_label = ast.literal_eval(row["label"])

        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            probs = torch.sigmoid(outputs.logits)[0].cpu().tolist()
            preds = (torch.tensor(probs) > THRESHOLD).int().tolist()

        true_text = label_to_text(true_label)
        pred_text = label_to_text(preds)

        title = (
            f"True: {true_text}\n"
            f"Pred: {pred_text}\n"
            f"P(person)={probs[0]:.2f}, P(car)={probs[1]:.2f}"
        )

        plt.subplot(rows, cols, idx)
        plt.imshow(image)
        plt.title(title, fontsize=10)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()