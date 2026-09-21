import torch
from torch.utils.data import DataLoader
from torchvision.datasets import GTSRB
from torchvision import transforms


def get_dataloaders(data_dir, batch_size=32):
    basic_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    train_dataset = GTSRB(
        root=data_dir,
        split="train",
        transform=basic_transform,
        download=True
    )
    test_dataset = GTSRB(
        root=data_dir,
        split="test",
        transform=basic_transform,
        download=True
    )    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader