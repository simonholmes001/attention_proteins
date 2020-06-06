import os

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import math

from protein_loader import ProteinLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = ProteinLoader('/home/the_beast/mathisi/simon/attention/data')

batch_size = 1
num_workers = 1
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

for features, labels in dataset:
    print(features)
    print(labels)
print(len(dataset))

print('=' * 83)

test_ds =  dataloader[:89, :, :]
valid_ds =  dataloader[89:, :, :]
print(test_ds.shape)
print(valid_ds.shape)

print('=' * 83)

"""
num_epochs = 10
total_samples = len(dataset)
n_iterations = math.ceil(total_samples / batch_size)
print(total_samples, n_iterations)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        print(f"epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {labels.shape}")

"""