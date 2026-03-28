# ViT + LoRA for Person/Car Multi-Label Classification

## 1. Project Overview

This project implements a **multi-label image classification pipeline** based on the COCO dataset, focusing on detecting whether an image contains:

* Person
* Car

The model is built on top of a pre-trained Vision Transformer (**ViT-base-patch16-224**) and fine-tuned efficiently using **LoRA (Low-Rank Adaptation)**.

The goal of this work is to:

* Build a complete training pipeline from raw data to model output
* Apply parameter-efficient fine-tuning (PEFT)
* Enable reproducible training and evaluation

---

## 2. Model Architecture

### Base Model

* Model: ViT-base-patch16-224
* Source: Hugging Face Transformers

### Fine-tuning Method

* Method: LoRA (via PEFT)
* Injected modules:

  * attention.query
  * attention.key
  * attention.value
  * attention.output.dense

### Task Type

* Multi-label classification (2 classes: person, car)

---

## 3. Dataset

### Source

* COCO 2017 dataset

### Data Format

We convert COCO annotations into CSV format:

```text
file_path,label
train2017/000000000009.jpg,"[1,0]"
train2017/000000000025.jpg,"[1,1]"
```

Where:

* `[1,0]` → person only
* `[0,1]` → car only
* `[1,1]` → both present

### Preprocessing

* Resize to 224×224
* Normalize using ImageNet statistics
* Convert to tensor

### Data Augmentation (training only)

* RandomResizedCrop
* Horizontal Flip

---

## 4. Project Structure

```text
week3/
│
├── src/
│   ├── train.py        # Main training script
│   ├── dataset.py      # Custom Dataset class
│   └── utils.py        # Utility functions (metrics, LoRA config)
│
├── outputs/
│   ├── models/         # Saved PEFT models
│   └── logs/           # Training logs
```

---

## 5. Training Pipeline

### Step 1: Load Processor & Model

* AutoImageProcessor
* AutoModelForImageClassification

### Step 2: Inject LoRA

Only a small number of parameters are trained:

* Rank (r) = 8
* Dropout = 0.1

### Step 3: Dataset & Dataloader

* Custom Dataset (CSV-based)
* PyTorch DataLoader

### Step 4: Training Setup

* Loss: BCEWithLogitsLoss (multi-label)
* Optimizer: AdamW
* Scheduler: Linear warmup + decay
* Mixed precision: FP16 (Accelerate)

### Step 5: Training Loop

* Forward pass
* Loss computation
* Backpropagation (Accelerator)
* Metrics computation (Precision / Recall / F1)

---

## 6. Evaluation Metrics

We use **micro-averaged metrics**:

* Precision
* Recall
* F1-score

Prediction is obtained via:

```text
sigmoid(logits) > 0.5
```

---

## 7. Model Saving

The best model (based on validation F1) is saved to:

```text
outputs/models/vit-base-patch16-224-lora-person-car/
```

Saved components include:

* LoRA adapter weights
* Adapter configuration
* Image processor configuration

---

## 8. How to Run

### 1. Prepare Environment

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Make sure COCO is located at:

```text
D:\Praktikum_DJi\coco\
```

And CSV files exist:

```text
D:\Praktikum_DJi\output\coco_person_car_train.csv
D:\Praktikum_DJi\output\coco_person_car_val.csv
```

### 3. Run Training

```bash
cd src
python train.py
```

---

## 9. Key Features

* Parameter-efficient fine-tuning (LoRA)
* Custom dataset pipeline from COCO
* Multi-label classification setup
* Mixed precision training (FP16)
* Modular and reproducible structure

---

## 10. Notes

* COCO dataset is not included due to size
* Paths may need adjustment depending on environment
* Model can be reloaded using PEFT for inference or further training

---

## 11. Future Improvements

* Add TensorBoard or Weights & Biases logging
* Improve data balancing
* Extend to more COCO categories
* Add inference script

---
