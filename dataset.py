"""
FashionMNIST & MNIST:
Classes | Training | Testing
10      | 60000    | 10000

EMNIST:
Split name | Classes | Training | Testing | Info
byclass    | 62      | 697932   | 116323  | Digits + uppercase and lowercase letters
balance    | 47      | 112800   | 18800   | Merged similar letters (e.g., 'C' and 'c', a recommended balanced subset)
letters    | 26      | 88800    | 14800   | Uppercase letters only (labeled A~Z)
"""

import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms

class CustomImageDataset(Dataset):
    """
    自定義資料集，讀取單一資料夾內的圖片 (001.png, 002.png ...)
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("L")  # 如果是灰階圖就用 "L"，彩色就 "RGB"
        if self.transform:
            image = self.transform(image)
        return image, 0   # 這裡沒有 label，就給個 dummy=0

def get_dataloaders(dataset_config):
    """
    生成訓練、驗證、測試的 DataLoader
    dataset_config (dict) : 包含以下設定
        - dataset_name: "MNIST" | "EMNIST" | "FashionMNIST"
        - batch_size
        - num_workers
        - valid_ratio
        - resize: tuple
        - augmentation: dict
    """
    dataset_name = dataset_config.get("dataset_name", "MNIST")
    aug_cfg = dataset_config.get("augmentation", {})

    # 建立 transform list
    transform_list = [
        transforms.Resize(dataset_config["resize"]),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]

    # 資料增強
    if aug_cfg.get("use_random_rotation", False):
        transform_list.append(transforms.RandomRotation(aug_cfg.get("rotation_degrees", 10)))
    if aug_cfg.get("use_random_affine", False):
        transform_list.append(
            transforms.RandomAffine(degrees=0, translate=aug_cfg.get("translate_ratio", (0.1, 0.1)))
        )

    transform = transforms.Compose(transform_list)

    # 選擇 dataset
    if dataset_name == "MNIST":
        full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == "FashionMNIST":
        full_train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == "EMNIST":
        split_name = dataset_config.get("emnist_split", "byclass")
        full_train_dataset = datasets.EMNIST(root='./data', split=split_name, train=True, download=True, transform=transform)
        test_dataset = datasets.EMNIST(root='./data', split=split_name, train=False, download=True, transform=transform)
    elif dataset_name == "Custom":
        # 自定義圖片資料集 (001.png, 002.png, ...)
        full_dataset = CustomImageDataset(root_dir=dataset_config["root"], transform=transform)

        # 切 train/valid/test
        total_size = len(full_dataset)
        test_size = int(total_size * dataset_config.get("test_ratio", 0.1))
        valid_size = int(total_size * dataset_config.get("valid_ratio", 0.1))
        train_size = total_size - valid_size - test_size
        train_dataset, valid_dataset, test_dataset = random_split(full_dataset, [train_size, valid_size, test_size])

        # DataLoader
        train_loader = DataLoader(train_dataset, batch_size=dataset_config["batch_size"], shuffle=True, num_workers=dataset_config["num_workers"])
        valid_loader = DataLoader(valid_dataset, batch_size=dataset_config["batch_size"], shuffle=False, num_workers=dataset_config["num_workers"])
        test_loader = DataLoader(test_dataset, batch_size=dataset_config["batch_size"], shuffle=False, num_workers=dataset_config["num_workers"])
        return train_loader, valid_loader, test_loader
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    # train/valid split
    valid_size = int(len(full_train_dataset) * dataset_config["valid_ratio"])
    train_size = len(full_train_dataset) - valid_size
    train_dataset, valid_dataset = random_split(full_train_dataset, [train_size, valid_size])

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=dataset_config["batch_size"], shuffle=True, num_workers=dataset_config["num_workers"])
    valid_loader = DataLoader(valid_dataset, batch_size=dataset_config["batch_size"], shuffle=False, num_workers=dataset_config["num_workers"])
    test_loader = DataLoader(test_dataset, batch_size=dataset_config["batch_size"], shuffle=False, num_workers=dataset_config["num_workers"])

    return train_loader, valid_loader, test_loader

if __name__ == "__main__":
    
    # 測試 Custom dataset
    DATASET_CONFIG = {
        "dataset_name": "Custom",
        "root": "./datasets/my_images",   # 放 001.png, 002.png ...
        "batch_size": 16,
        "num_workers": 0,
        "valid_ratio": 0.1,
        "test_ratio": 0.1,
        "resize": (128, 128),
        "augmentation": {
            "use_random_rotation": False,
            "rotation_degrees": 10,
            "use_random_affine": False,
            "translate_ratio": (0.1, 0.1)
        }
    }# 測試用
    DATASET_CONFIG = {
        "dataset_name": "EMNIST",
        "emnist_split": "byclass",
        "batch_size": 64,
        "num_workers": 0,
        "valid_ratio": 0.1,
        "resize": (128, 128),
        "augmentation": {
            "use_random_rotation": True,
            "rotation_degrees": 10,
            "use_random_affine": False,
            "translate_ratio": (0.1, 0.1)
        }
    }

    train_loader, valid_loader, test_loader = get_dataloaders(DATASET_CONFIG)
    print(f"Train: {len(train_loader.dataset)}, Valid: {len(valid_loader.dataset)}, Test: {len(test_loader.dataset)}")
