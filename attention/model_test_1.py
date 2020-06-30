import torch
from torch.utils.data import Dataset, DataLoader, random_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from protein_loader import ProteinLoader

x = ProteinLoader('/media/the_beast/A/mathisi_tests/data/LSTM/data')
dataset = x.get_all_data()
N = 64
n_workers = 5
validation_split = 0.01
shuffle = True

protein_loader = DataLoader(dataset=dataset, batch_size=N, shuffle=True, num_workers=n_workers)


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H2)
        self.linear3 = torch.nn.Linear(H2, H3)
        self.linear4 = torch.nn.Linear(H3, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu = self.linear1(x).clamp(min=0) # https://pytorch.org/docs/master/generated/torch.clamp.html
        # My idea: writing h_relu like this means that one "clamps" the output of the linear
        # layer between 0 (when x < 0) and x (when x > 0), effectively, the relu function
        h_relu = self.linear2(h_relu).clamp(min=0)
        h_relu = self.linear3(h_relu).clamp(min=0)
        y_pred = self.linear4(h_relu)
        return y_pred


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
# N = 64
D_in, H, D_out = 17400, 1000, 300
H2 = 750
H3 = 500

# Create random Tensors to hold inputs and outputs
# x = torch.randn(N, D_in)
# y = torch.randn(N, D_out)

# Construct our model by instantiating the class defined above
model = TwoLayerNet(D_in, H, D_out)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 1000
for t in range(num_epochs):
    for idx, (x, y) in enumerate(protein_loader):
        x = x.to(device=device)
        y = y.to(device=device)
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x).to(device=device)

    # Compute and print loss
    loss = criterion(y_pred, y)
    if t % 100 == 99:
        print(f"Epoch {t+1}/{num_epochs}: loss: {loss.item()}")
    if t == num_epochs - 1:
        print(f"y_pred: {y_pred}\n"
                f"y: {y}")

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()