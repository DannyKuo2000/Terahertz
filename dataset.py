import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from config import DATASET_CONFIG

"""
Resize到128，跟學長的方法一樣
"""
### download dataset
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


import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

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
    # 測試用
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
