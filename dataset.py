import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


"""
Resize到128，跟學長的方法一樣
"""

def get_dataloaders(batch_size=64, num_workers=0):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # resize from 28 to 128
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader