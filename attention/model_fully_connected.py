import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchtext.data import Field, BucketIterator
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np

import random
import math
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from protein_loader import ProteinLoader

x = ProteinLoader('/media/the_beast/A/mathisi_tests/data/LSTM/data')
dataset = x.get_all_data()
batch_size = 10
n_workers = 1
validation_split = 0.01
shuffle = True

protein_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)
print(f"Dataset size: {len(dataset)}\n"
      f"ProteinLoader size: {len(protein_loader)}")

# Hyperparametres
input_size = 17400
hidden_size = 512*2
hidden_size_2 = 512
hidden_size_3 = 256
num_layers = 5
dropout = 0.1
out_features = 300
num_epochs = 50
learning_rate = 0.01

# Create model, see https://www.youtube.com/watch?v=jGst43P-TJA
class Model(nn.Module):

    def __init__(self, input_size, num_classes):
        super(Model, self).__init__()
        # self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
        #                      bias=True, batch_first=True, dropout=dropout, bidirectional=True)
        self.input_size = input_size
        self.num_classes = num_classes
        self.fc1 = nn.Linear(in_features=input_size, out_features=hidden_size, bias=True)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)
        self.fc3 = nn.Linear(in_features=hidden_size, out_features=hidden_size_2, bias=True)
        self.fc4 = nn.Linear(in_features=hidden_size_2, out_features=hidden_size_3, bias=True)
        self.fc5 = nn.Linear(in_features=hidden_size_3, out_features=out_features, bias=True)

    def forward(self, x):
        # h0 = torch.zeros(num_layers*2, x.size(0), hidden_size).to(device)
        # c0 = torch.zeros(num_layers*2, x.size(0), hidden_size).to(device)
        # output, _ = self.lstm1(x, (h0, c0))
        # output = self.fc1(ouput[:, -1, :])

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.softmax(self.fc5(x), dim=1)

        return x

# Test that model generates output tensor of correct shape
# model = Model()
# print(model)
# x = torch.rand(batch_size, input_size)
# print(model(x).shape)

# Initialise network
model = Model(input_size=input_size, num_classes=out_features).to(device)

# Loss and optimiser
criterion = nn.MSELoss()
optimiser = optim.SGD(model.parameters(), lr=learning_rate)

# Train network
z = 0
for epoch in range(num_epochs):
    print(f"Epoch: {epoch+1}/{num_epochs}")
    for idx, (data, targets) in enumerate(protein_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # print(data.shape)

        # forward pass
        scores = model(data)
        loss = criterion(scores, targets)
        print(f"{z+1}")
        print(f"Loss is: {loss}")
        z += 1

        # backward pass
        optimiser.zero_grad()
        loss.backward()

        # gradient descent
        optimiser.step() # update weights

# Check accuracy
# def check_accuracy(loader, model):
#     num_correct = 0
#     num_samples = 0
#     model.eval()
#
#     with torch.no_grad():
#         for x, y in loader:
#             x = x.to(device=device)
#             y = y.to(device=device)
#
#             scores = model(x)
#             _, predictions = scores.max(1)
#             num_correct += (predictions == y).sum()
#             num_samples += predictions.size(0)
#
#         accuracy = float(num_correct) / float(num_samples)
#         print(accuracy)
#
#     model.train()
#
# check_accuracy(protein_loader, model)