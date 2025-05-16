import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

"""
Resize到128，跟學長的方法一樣
"""

def get_dataloaders(batch_size=64, num_workers=0, valid_ratio=0.1):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # resize from 28 to 128
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),

        ### 較不常使用的augmentation
        #transforms.RandomRotation(10)
        #transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
    ])

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
    full_train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    #full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    #test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    full_train_dataset = datasets.EMNIST(root='./data', split='byclass', train=True, download=True, transform=transform)
    test_dataset = datasets.EMNIST(root='./data', split='byclass', train=False, download=True, transform=transform)


    ### Spliting training, validation and testing dataset
    valid_size = int(len(full_train_dataset) * valid_ratio)
    train_size = len(full_train_dataset) - valid_size
    train_dataset, valid_dataset = random_split(full_train_dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader

if __name__ == "__main__":
    get_dataloaders()