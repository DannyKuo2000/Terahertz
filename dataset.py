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
import time
from PIL import Image
import torch
import random
from torch.utils.data import DataLoader, random_split, Dataset, Subset, ConcatDataset
from torchvision import datasets, transforms
from config import DATASET_CONFIG

# ====== 平行化 ======
from torch.utils.data.distributed import DistributedSampler

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

def get_dataloaders(dataset_config, per_gpu_batch, num_workers, distributed):
    """
    生成訓練、驗證、測試的 DataLoader
    dataset_config (dict):
        - dataset_name: "MNIST" | "EMNIST" | "FashionMNIST" | "MNIST+EMNIST" | "Custom"
        - batch_size
        - num_workers
        - valid_ratio
        - test_ratio (for Custom)
        - resize: tuple
        - center_crop: tuple
        - augmentation: dict
        - seed: int (optional)
    """

    # ====== 隨機種子 ======
    seed = dataset_config.get("seed", 42)
    if seed == "random":  # 允許自動亂數種子
        seed = int(time.time()) % (2**32)
    print(f"[Dataset] Using seed = {seed}")
    torch.manual_seed(seed)
    random.seed(seed)

    # ====== augmentation & transform ======
    aug_cfg = dataset_config.get("augmentation", {})
    transform_list = [
        transforms.Resize(dataset_config["resize"]),
        transforms.CenterCrop(dataset_config["center_crop"]),
        transforms.ToTensor(),  # from 0~255 to 0~1
        # transforms.Normalize((0.5,), (0.5,)) # No normalization here, it will distroy the E field simulation
    ]
    if aug_cfg.get("use_random_rotation", False):
        transform_list.append(
            transforms.RandomRotation(aug_cfg.get("rotation_degrees", 10))
        )
    if aug_cfg.get("use_random_affine", False):
        transform_list.append(
            transforms.RandomAffine(degrees=0, translate=aug_cfg.get("translate_ratio", (0.1, 0.1)))
        )
    transform = transforms.Compose(transform_list)

    dataset_name = dataset_config.get("dataset_name")

    # ====== MNIST ======
    if dataset_name == "MNIST":
        full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # ====== FashionMNIST ======
    elif dataset_name == "FashionMNIST":
        full_train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # ====== EMNIST ======
    elif dataset_name == "EMNIST":
        split_name = dataset_config.get("emnist_split", "byclass")
        full_train_dataset = datasets.EMNIST(root='./data', split=split_name, train=True, download=True, transform=transform)
        test_dataset = datasets.EMNIST(root='./data', split=split_name, train=False, download=True, transform=transform)

    # ====== MNIST + EMNIST ======
    elif dataset_name == "MNIST+EMNIST":
        # MNIST 全部
        mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        # EMNIST (subset 取 25%)
        split_name = dataset_config.get("emnist_split", "byclass")
        emnist_train = datasets.EMNIST(root='./data', split=split_name, train=True, download=True, transform=transform)
        emnist_test = datasets.EMNIST(root='./data', split=split_name, train=False, download=True, transform=transform)

        # 按 MNIST 的數量比例取 25%
        emnist_train_size = int(len(mnist_train) * 0.25)
        emnist_test_size = int(len(mnist_test) * 0.25)

        emnist_train_indices = random.sample(range(len(emnist_train)), emnist_train_size)
        emnist_test_indices = random.sample(range(len(emnist_test)), emnist_test_size)

        emnist_train_subset = Subset(emnist_train, emnist_train_indices)
        emnist_test_subset = Subset(emnist_test, emnist_test_indices)

        # 合併成混合 dataset
        full_train_dataset = ConcatDataset([mnist_train, emnist_train_subset])
        test_dataset = ConcatDataset([mnist_test, emnist_test_subset])

    # ====== Custom dataset ======
    elif dataset_name == "Custom":
        full_dataset = CustomImageDataset(root_dir=dataset_config["root"], transform=transform)
        total_size = len(full_dataset)
        test_size = int(total_size * dataset_config.get("test_ratio", 0.1))
        valid_size = int(total_size * dataset_config.get("valid_ratio", 0.1))
        train_size = total_size - valid_size - test_size

        train_dataset, valid_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, valid_size, test_size],
            generator=torch.Generator().manual_seed(seed)
        )
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    # ====== train/valid split ======
    valid_size = int(len(full_train_dataset) * dataset_config["valid_ratio"])
    train_size = len(full_train_dataset) - valid_size

    train_dataset, valid_dataset = random_split(
        full_train_dataset,
        [train_size, valid_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # ====== DataLoader ======
    """train_loader = DataLoader(train_dataset, batch_size=dataset_config["batch_size"], shuffle=True, num_workers=dataset_config["num_workers"])
    valid_loader = DataLoader(valid_dataset, batch_size=dataset_config["batch_size"], shuffle=False, num_workers=dataset_config["num_workers"])
    test_loader = DataLoader(test_dataset, batch_size=dataset_config["batch_size"], shuffle=False, num_workers=dataset_config["num_workers"])"""

    # ====== Sampler 設定 ======
    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
        test_sampler  = DistributedSampler(test_dataset,  shuffle=False)
    else:
        train_sampler = None
        valid_sampler = None
        test_sampler  = None

    # ====== DataLoader 建立 ======
    train_loader = DataLoader(
        train_dataset,
        batch_size=per_gpu_batch,
        shuffle=(train_sampler is None),   # 單卡用 shuffle、多卡交給 sampler
        num_workers=num_workers,
        sampler=train_sampler,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=per_gpu_batch,
        shuffle=False,
        num_workers=num_workers,
        sampler=valid_sampler,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=per_gpu_batch,
        shuffle=False,
        num_workers=num_workers,
        sampler=test_sampler,
        pin_memory=True,
    )

    # ====== Visualization ======
    print(f"[Dataset] Train size = {len(train_dataset)} | Valid size = {len(valid_dataset)} | Test size = {len(test_dataset)}")
    
    return train_loader, valid_loader, test_loader

# 測試
if __name__ == "__main__":
    train_loader, valid_loader, test_loader = get_dataloaders(DATASET_CONFIG)
    print(f"Train: {len(train_loader.dataset)}, Valid: {len(valid_loader.dataset)}, Test: {len(test_loader.dataset)}")
