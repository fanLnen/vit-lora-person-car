# 基于 ViT + LoRA 的多标签图像分类（COCO：Person & Car）

基于 **Vision Transformer (ViT)**，结合 **LoRA（Low-Rank Adaptation）参数高效微调方法**，实现了一个**多标签图像分类模型**。

模型在 **COCO 数据集** 上进行训练，仅关注两个类别：

* **Person（人）**
* **Car（汽车）**

目标：模型输入一张图片，输出图中是否存在这两个目标。

---

##  任务定义（使用第一周整理的关于图片的CSV文件）

本任务属于 **多标签分类（Multi-Label Classification）**，即每张图片可以同时包含多个类别：

|标签       | 含义        |
| -------- | ------      |
| `[1, 0]` | 有人，无车   |
| `[0, 1]` | 无人，有车   |
| `[1, 1]` | 同时人和车   |
| `[0, 0]` | 都没有       |

---

##  模型结构

* Backbone：预训练的`ViT-base-patch16-224`
* 微调方式：LoRA（PEFT）
* 输出维度：2（person / car）
* 损失函数：`BCEWithLogitsLoss`

---

##  LoRA 注入位置

LoRA 被注入到 Transformer 的注意力层中：

```
target_modules = [
    "query",
    "key",
    "value",
    "output.dense"
]
```

模型参数情况：

```
可训练参数 ≈ 0.96M
总参数     ≈ 86.7M
训练比例   ≈ 1.1%
```


##  结构
.
├── train_vit_lora_multilabel.py   # 训练脚本
├── evaluate_vit_lora.py           # 模型评估（Precision / Recall / F1）
├── inference_vit_lora.py          # 单张图片推理
├── visualize_predictions.py       # 可视化预测结果
├── output/
│   └── checkpoints/
│       ├── best_vit_lora_person_car.pt
│       └── last_vit_lora_person_car.pt

---

##  各脚本说明

### 1️ train_vit_lora_multilabel.py

功能：

* 加载 COCO CSV 数据
* 构建 Dataset & DataLoader
* 加载 ViT 模型
* 注入 LoRA
* 进行训练（多 epoch）
* 保存模型：


best_vit_lora_person_car.pt   # 验证集最优模型（推荐使用）
last_vit_lora_person_car.pt   # 最后一轮模型


---

### 2 evaluate_vit_lora.py

功能：

* 在验证集上评估模型性能
* 输出指标：


Micro Precision / Recall / F1
Per-class Precision / Recall / F1
Macro F1

---

### 3️ inference_vit_lora.py

功能：

* 对单张图片进行预测
* 输出：

```
logits
probabilities
predictions（0/1）
```

---

### 4️ visualize_predictions.py

功能：

* 可视化模型预测结果
* 在图片上显示：

```
预测标签 + 概率
```

用于直观分析模型效果。

---

## 结果

### 🔹 Overall Metrics（整体表现）

```
Micro Precision: 0.9599
Micro Recall:    0.9501
Micro F1:        0.9550
```

 模型整体表现优秀（F1 ≈ 0.955）

---

###  分类别指标

#### Person 类

```
Precision: 0.9779
Recall:    0.9851
F1:        0.9815
```

 几乎完美识别人类目标

---

#### Car 类

```
Precision: 0.8589
Recall:    0.7738
F1:        0.8142
```

 存在一定问题：
* Recall 较低（漏检较多）
* 对 car 的识别能力弱于 person

---

### 🔹 Macro F1

```
Macro F1: 0.8978
```

 平衡考虑各类别后的整体表现

---

##  模型优缺点分析

###  优点

* 使用 LoRA，仅训练约 1% 参数，效率高
* Person 类识别效果极好（F1 ≈ 0.98）
* 整体 F1 达到 0.95，性能稳定
* 完整实现训练 → 评估 → 推理 → 可视化 pipeline

---

###  不足

* Car 类 Recall 较低（漏检较多）
* 数据存在类别不平衡（person ≫ car）
* 模型为分类模型，无法进行目标定位（无 bounding box）

---

##  后续优化方向

* 调整 BCE loss 的 `pos_weight` 解决类别不平衡
* 调整 car 类阈值（如 0.5 → 0.4）提高 Recall
* 增加训练数据或数据增强
* 扩展更多类别（COCO 多类）
* 升级为目标检测模型（如 YOLO / DETR）

---

##  总结

本项目完成了一个完整的计算机视觉深度学习流程：

数据处理 → 模型构建 → LoRA 微调 → 模型训练 → 性能评估 → 推理与可视化


并成功实现：

 基于 ViT + LoRA 的多标签图像分类模型
 在 COCO 数据集上达到较高性能（F1 ≈ 0.95）

---

##  依赖环境

torch
transformers
peft
pandas
Pillow



