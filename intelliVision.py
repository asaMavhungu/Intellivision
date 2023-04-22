import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


# MNIST dataset 
training_data = datasets.CIFAR10(
    root='./data', 
    train=True, 
    transform=ToTensor(),  
    download=True
    )

test_data = datasets.CIFAR10(
    root='./data',                         
    train=False, 
    transform=ToTensor(),
    download=True
    )