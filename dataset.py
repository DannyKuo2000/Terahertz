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
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, random_split, Dataset, Subset, ConcatDataset
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler
from PIL import Image


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted(
            [f for f in os.listdir(root_dir) if f.endswith((".png", ".jpg"))]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, 0


def get_dataloaders(dataset_config, per_gpu_batch, num_workers, distributed):

    seed = dataset_config.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)

    transform_list = []
    if dataset_config["resize"] is not None:
        transform_list.append(transforms.Resize(dataset_config["resize"]))
    if dataset_config["center_crop"] is not None:
        transform_list.append(transforms.CenterCrop(dataset_config["center_crop"]))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)

    name = dataset_config["dataset_name"]

    # ================= dataset =================
    if name == "MNIST":
        full = datasets.MNIST("./data", True, download=True, transform=transform)
        test = datasets.MNIST("./data", False, download=True, transform=transform)

    elif name == "FashionMNIST":
        full = datasets.FashionMNIST("./data", True, download=True, transform=transform)
        test = datasets.FashionMNIST("./data", False, download=True, transform=transform)

    elif name == "EMNIST":
        full = datasets.EMNIST("./data", split=dataset_config["emnist_split"], train=True, download=True, transform=transform)
        test = datasets.EMNIST("./data", split=dataset_config["emnist_split"], train=False, download=True, transform=transform)

    elif name == "Custom":
        full = CustomImageDataset(dataset_config["root"], transform)
        test = full

    elif name == "MNIST+EMNIST":
        seed = dataset_config.get("seed", 42)
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        # -------------------------
        # load full datasets
        # -------------------------
        mnist = datasets.MNIST(
            "./data",
            train=True,
            download=True,
            transform=transform
        )

        emnist = datasets.EMNIST(
            "./data",
            split=dataset_config["emnist_split"],
            train=True,
            download=True,
            transform=transform
        )

        # -------------------------
        # target sizes (IMPORTANT)
        # -------------------------
        mnist_size = dataset_config.get("mnist_size", 10000)
        emnist_size = dataset_config.get("emnist_size", 60000)

        # safety clamp
        mnist_size = min(mnist_size, len(mnist))
        emnist_size = min(emnist_size, len(emnist))

        # -------------------------
        # reproducible sampling
        # -------------------------
        mnist_indices = np.random.choice(
            len(mnist),
            mnist_size,
            replace=False
        )

        emnist_indices = np.random.choice(
            len(emnist),
            emnist_size,
            replace=False
        )

        mnist_subset = Subset(mnist, mnist_indices)
        emnist_subset = Subset(emnist, emnist_indices)

        # -------------------------
        # final dataset
        # -------------------------
        full = ConcatDataset([mnist_subset, emnist_subset])

        # test set (fixed official MNIST test)
        mnist_test = datasets.MNIST(
            "./data",
            train=False,
            download=True,
            transform=transform
        )
        emnist_test = datasets.EMNIST(
            "./data",
            split=dataset_config["emnist_split"],
            train=False,
            download=True,
            transform=transform
        )
        emnist_test_indices = np.random.choice(
            len(emnist_test),
            int(len(mnist_test) * dataset_config["emnist_test_ratio"]),
            replace=False
        )
        emnist_test_subset = Subset(emnist_test, emnist_test_indices)
        test = ConcatDataset([mnist_test, emnist_test_subset])
    else:
        raise ValueError(name)

    # ================= split =================
    val_size = int(len(full) * dataset_config["valid_ratio"])
    train_size = len(full) - val_size

    train, val = random_split(full, [train_size, val_size])

    # ================= sampler =================
    train_sampler = DistributedSampler(train, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train,
        batch_size=per_gpu_batch,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val,
        batch_size=per_gpu_batch,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test,
        batch_size=per_gpu_batch,
        shuffle=False,
        num_workers=num_workers,
    )

    print(
        f"[Dataset] train={len(train)} val={len(val)} test={len(test)}"
    )

    return train_loader, val_loader, test_loader
