import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader

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
hidden_size = 256
num_layers = 2
dropout = 0.1
out_features = 300
num_epochs = 2
learning_rate = 0.01

# Create model, see https://www.youtube.com/watch?v=jGst43P-TJA
class Model(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=False)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=device)

        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)

        return out

# Test that model generates output tensor of correct shape
# model = Model()
# print(model)
# x = torch.rand(batch_size, input_size)
# print(model(x).shape)

# Initialise network
model = Model(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=out_features).to(device)

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

        data = data.reshape(data.shape[0], 1)

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