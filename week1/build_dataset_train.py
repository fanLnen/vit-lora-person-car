from collections import defaultdict, Counter
from pycocotools.coco import COCO
import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

ann_file = r"coco\annotations\instances_train2017.json"

coco = COCO(ann_file)

# 1. 找类别 id
person_id = coco.getCatIds(catNms=["person"])[0]
car_id = coco.getCatIds(catNms=["car"])[0]

print("person category id:", person_id)
print("car category id:", car_id)

# 2. 只取 person 和 car 的所有标注
ann_ids = coco.getAnnIds(catIds=[person_id, car_id], iscrowd=None)
anns = coco.loadAnns(ann_ids)

# 3. 为每张图生成多标签 [is_person, is_car]
image_id_to_label = defaultdict(lambda: [0, 0])

for ann in anns:
    image_id = ann["image_id"]
    cat_id = ann["category_id"]

    if cat_id == person_id:
        image_id_to_label[image_id][0] = 1
    elif cat_id == car_id:
        image_id_to_label[image_id][1] = 1

# 4. 统计实例数量
person_instance_count = sum(1 for ann in anns if ann["category_id"] == person_id)
car_instance_count = sum(1 for ann in anns if ann["category_id"] == car_id)

print("person instance count:", person_instance_count)
print("car instance count:", car_instance_count)

# 5. 统计图像标签分布
label_counter = Counter()

for label in image_id_to_label.values():
    label_counter[tuple(label)] += 1

only_person = label_counter[(1, 0)]
only_car = label_counter[(0, 1)]
both = label_counter[(1, 1)]
total_valid = len(image_id_to_label)

print("only person images:", only_person)
print("only car images:", only_car)
print("person + car images:", both)
print("total valid images:", total_valid)

print("check sum:", only_person + only_car + both)


print("=========================================================")
print("=========================================================")


# 图片目录
img_dir = r"coco\train2017"

# 可视化输出目录
vis_dir = r"output\visualizations"
os.makedirs(vis_dir, exist_ok=True)

# 随机选择 5 张有效图片
sample_image_ids = random.sample(list(image_id_to_label.keys()), 5)

for image_id in sample_image_ids:

    img_info = coco.loadImgs(image_id)[0]
    img_path = os.path.join(img_dir, img_info["file_name"])

    image = Image.open(img_path)

    plt.figure(figsize=(8,6))
    plt.imshow(image)
    plt.axis("off")

    # 找该图片的person/car标注
    ann_ids = coco.getAnnIds(imgIds=[image_id], catIds=[person_id, car_id])
    anns = coco.loadAnns(ann_ids)

    ax = plt.gca()

    for ann in anns:

        x, y, w, h = ann["bbox"]

        rect = plt.Rectangle(
            (x, y),
            w,
            h,
            fill=False,
            edgecolor="red",
            linewidth=2
        )

        ax.add_patch(rect)

        label = "person" if ann["category_id"] == person_id else "car"

        ax.text(
            x,
            y,
            label,
            color="yellow",
            fontsize=12,
            bbox=dict(facecolor="black", alpha=0.5)
        )

    save_path = os.path.join(vis_dir, img_info["file_name"])
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

print("Visualization images saved to:", vis_dir)


print("=========================================================")
print("=========================================================")


dataset = []

for image_id, label in image_id_to_label.items():

    img_info = coco.loadImgs(image_id)[0]

    file_name = img_info["file_name"]

    file_path = os.path.join("train2017", file_name)

    dataset.append({
        "file_path": file_path,
        "label": label
    })

df = pd.DataFrame(dataset)

csv_path = r"output\coco_person_car_train.csv"

df.to_csv(csv_path, index=False)

print("CSV saved to:", csv_path)