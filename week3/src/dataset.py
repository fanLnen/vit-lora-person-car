import os
import ast
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CocoPersonCarMultiLabelDataset(Dataset):
    """
    Multi-label dataset for person/car classification.

    Expected CSV columns:
    - file_path : relative path, e.g. train2017/000000000009.jpg
    - label     : string representation of python list, e.g. "[1, 0]" or "[1, 1]"
    """

    def __init__(self, csv_file: str, image_root: str, processor, train: bool = False):
        self.df = pd.read_csv(csv_file)
        self.image_root = image_root
        self.processor = processor
        self.train = train

        # Data augmentation only for training
        if self.train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
            ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        relative_path = row["file_path"]
        image_path = os.path.join(self.image_root, relative_path)

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # Processor handles normalization / tensor conversion
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)

        label_list = ast.literal_eval(row["label"])
        labels = torch.tensor(label_list, dtype=torch.float)

        return {
            "pixel_values": pixel_values,
            "labels": labels
        }