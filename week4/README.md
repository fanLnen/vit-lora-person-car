把下面这份直接整体替换进你的 `README.md` 就行，GitHub 显示会正常。

````markdown
# Week4: ViT + LoRA Training Pipeline with Validation and Evaluation

## 1. Project Goal

This week’s work continues from Week 3 and focuses on completing the full training workflow. The main tasks are:

- implement the complete training loop
  - forward pass
  - loss computation
  - backward propagation
  - optimizer step
- implement the validation phase
  - validation loss computation
  - validation metric computation
  - saving the best model
- improve the evaluation utility module
  - multi-label classification metrics
  - single-label classification metrics
- run at least one epoch and perform validation

Compared with Week 3, this week mainly improves code completeness, metric coverage, and mode flexibility.

## 2. Completed Work

### 2.1 Dataset Module

The dataset module has been extended to support both:

- `multilabel`
- `singlelabel`

In `multilabel` mode:
- the `label` column in the CSV file must be a Python-style list string, such as:
  - `[1, 0]`
  - `[0, 1]`
  - `[1, 1]`

In `singlelabel` mode:
- the `label` column in the CSV file must be a single integer class id, such as:
  - `0`
  - `1`

The dataset module also supports:
- training augmentation
- validation preprocessing
- image loading from relative paths
- processor-based normalization and tensor conversion

### 2.2 Training Script

The training script now supports the full pipeline:

- load image processor
- load pretrained ViT model
- inject LoRA modules
- create training and validation datasets
- create DataLoaders
- choose loss function according to mode
- run training loop
- run validation loop
- compute metrics
- save the best model according to validation F1

### 2.3 Evaluation Utilities

The utility module now supports both task types.

For `multilabel` classification:
- Accuracy
- Precision
- Recall
- F1

For `singlelabel` classification:
- sample-level Accuracy
- per-class Precision
- per-class Recall
- Macro F1

## 3. Folder Structure

Recommended structure for this week:

```text
week4/
├── train.py
├── dataset.py
├── utils.py
````

After running the training script, the output directories will be created automatically in the parent folder:

```text
outputs/
├── logs/
│   ├── train_log_multilabel.txt
│   └── train_log_singlelabel.txt
└── models/
    └── vit-base-patch16-224-lora-person-car/
```

## 4. Requirements

Recommended environment: existing conda environment.

Main dependencies:

* python
* torch
* torchvision
* transformers
* peft
* accelerate
* pandas
* pillow
* tqdm

## 5. Data Format

### 5.1 Multi-label Mode

This mode is suitable when one image may contain both `person` and `car`.

Example CSV:

```csv
file_path,label
train2017/000000000009.jpg,"[1, 0]"
train2017/000000000025.jpg,"[0, 1]"
train2017/000000000030.jpg,"[1, 1]"
```

Meaning:

* first position = `person`
* second position = `car`

Examples:

* `[1, 0]` = person only
* `[0, 1]` = car only
* `[1, 1]` = both person and car

### 5.2 Single-label Mode

This mode is only suitable when each image belongs to exactly one class.

Example CSV:

```csv
file_path,label
train2017/000000000009.jpg,0
train2017/000000000025.jpg,1
```

For example:

* `0 = person`
* `1 = car`

## 6. Conditions for Using `singlelabel`

`singlelabel` cannot be directly used with the original person/car multi-label CSV.

This is because in the COCO person/car task, one image may contain both `person` and `car`, so the original labels are usually:

* `[1, 0]`
* `[0, 1]`
* `[1, 1]`

This is naturally a multi-label classification problem, so `multilabel` is the correct mode for the original dataset.

You should use `singlelabel` only if all of the following conditions are satisfied:

* you have created a separate single-label CSV file
* each image belongs to only one class
* the `label` column contains a single integer
* no image belongs to multiple classes at the same time

Therefore:

* use `multilabel` for the current original person/car task
* use `singlelabel` only when the dataset has been reorganized into a single-class classification task

## 7. How to Run

### 7.1 Multi-label Mode

If your CSV labels are list-style labels such as `[1, 0]` or `[1, 1]`, run:

```bash
python train.py --mode multilabel
```

### 7.2 Single-label Mode

If your CSV labels are integer class ids such as `0` or `1`, run:

```bash
python train.py --mode singlelabel
```

## 8. Outputs

After training, the script will generate the following outputs.

### 8.1 Console Output

For each epoch, the script prints:

* Train Loss
* Train Accuracy
* Train Precision
* Train Recall
* Train F1
* Validation Loss
* Validation Accuracy
* Validation Precision
* Validation Recall
* Validation F1

In `singlelabel` mode, per-class metrics will also be printed.

### 8.2 Log Files

Log files are saved in:

```text
outputs/logs/
```

Examples:

* `train_log_multilabel.txt`
* `train_log_singlelabel.txt`

### 8.3 Best Model

The best model based on validation F1 is saved in:

```text
outputs/models/vit-base-patch16-224-lora-person-car/
```

The saved files include:

* LoRA adapter weights
* PEFT configuration
* image processor configuration

```
```
