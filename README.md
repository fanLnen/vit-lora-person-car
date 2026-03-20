# ViT + LoRA Person vs Car Classification

## 1. Project Description

This project fine-tunes a Vision Transformer (ViT) using LoRA for binary classification (person vs car) based on the COCO dataset.

---

## 2. Environment Setup

Install dependencies:

```
pip install -r requirements.txt
```

---

## 3. Dataset

This project uses the COCO dataset (not included in the repository).

Download from:
https://cocodataset.org/#download

Required files:

* train2017/
* val2017/
* annotations/

---

## 4. Folder Structure

Place the dataset as follows:

```
D:/Praktikum_DJi/
│
├── coco/
│   ├── annotations/
│   ├── train2017/
│   └── val2017/
│
├── output/
├── week1/
├── week2/
```

---

## 5. Running Order

Run scripts in order:

```
cd week1
python xxx.py

cd ../week2
python xxx.py
```

---

## 6. Notes

* Dataset is not included due to size limitations
* Please ensure correct relative paths in the code

```
```
