import os
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from peft import LoraConfig, get_peft_model



# 1. 配置

MODEL_NAME = "google/vit-base-patch16-224"
CHECKPOINT_PATH = r"D:\Praktikum_DJi\output\checkpoints\best_vit_lora_person_car.pt"

#  测试的图片
IMAGE_PATH = r"D:\Praktikum_DJi\coco\val2017\000000508602.jpg"

THRESHOLD = 0.5
CLASS_NAMES = ["person", "car"]



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



# 3. 推理

def predict_one_image(image_path):
    processor, model = build_model()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits
        probs = torch.sigmoid(logits)
        preds = (probs > THRESHOLD).int()

    probs = probs[0].cpu().tolist()
    preds = preds[0].cpu().tolist()

    print(f"Image: {image_path}")
    print("\n===== Prediction Result =====")
    for class_name, prob, pred in zip(CLASS_NAMES, probs, preds):
        print(f"{class_name:>6} | probability = {prob:.4f} | prediction = {pred}")

    print(f"\nFinal multi-label prediction: {preds}")


if __name__ == "__main__":
    predict_one_image(IMAGE_PATH)