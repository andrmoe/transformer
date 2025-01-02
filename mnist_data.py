import numpy as np
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import pickle
import os
from torch.utils.data import DataLoader, TensorDataset


def make_training_data_loader():
    if os.path.exists("train_data"):
        with open("train_data", "rb") as f:
            train_data = pickle.load(f)
    else:
        train_data = datasets.MNIST(
            root="data",
            train=True,
            transform=ToTensor(),
            download=True
        )
        with open("train_data", "wb") as f:
            pickle.dump(train_data, f)
    return DataLoader(train_data, batch_size=100, shuffle=True, num_workers=1)


def make_test_data_loader():
    if os.path.exists("test_data"):
        with open("test_data", "rb") as f:
            test_data = pickle.load(f)
    else:
        test_data = datasets.MNIST(
            root="data",
            train=False,
            transform=ToTensor(),
            download=True
        )
        with open("test_data", "wb") as f:
            pickle.dump(test_data, f)

    return DataLoader(test_data, batch_size=100, shuffle=True, num_workers=1)


def manual_training_dataloader():
    if os.path.exists("train_data"):
        with open("train_data", "rb") as f:
            train_data = pickle.load(f)
    else:
        train_data = datasets.MNIST(
            root="data",
            train=True,
            transform=ToTensor(),
            download=True
        )
    data = train_data.data.numpy()
    data = data.reshape(data.shape[0], 1, data.shape[1], data.shape[2])
    targets = train_data.targets.numpy()

    dataset = TensorDataset(torch.tensor(data, dtype=torch.float32), torch.tensor(targets, dtype=torch.long))
    return DataLoader(dataset, batch_size=100, shuffle=True, num_workers=1)
