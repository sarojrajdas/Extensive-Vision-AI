import torch
from torchvision import datasets

def download_train_data(train_transforms):
    # Download Train Dataset
    train_data = datasets.MNIST(
        './data', 
        train=True, 
        download=True, 
        transform=train_transforms)
    return train_data


def download_test_data(test_transforms):
    # Download Test Dataset
    test_data = datasets.MNIST(
        './data', 
        train=False, 
        download=True, 
        transform=test_transforms
        )
    return test_data

def load_train_data(train_data, **dataloader_args):
    # Load Train Data
    train_loader = torch.utils.data.DataLoader(
        train_data,
        dataloader_args
    )
    return train_loader

def load_test_data(test_data, **dataloader_args):
    # Load Test Data
    test_loader = torch.utils.data.DataLoader(
        test_data, 
        dataloader_args
    )
    return test_loader
