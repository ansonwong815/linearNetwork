import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import time

data_size = 2000
inputs = np.random.rand(data_size, 3)


def f(x):
    return (x[:, 0] ** x[:, 1] + x[:, 1] ** x[:, 2]) ** 2


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 50),
            nn.ReLU(),
            nn.Linear(50, 30),
            nn.ReLU(),
            nn.Linear(30, 1))

    def forward(self, x):
        output = self.network(x)
        return output


class Dataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = torch.tensor(inputs).float()
        self.outputs = torch.tensor(outputs).float()

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


labels = f(inputs)
labels.resize(data_size, 1)
dataset = Dataset(inputs, labels)
train_data, test_data = torch.utils.data.random_split(dataset, [int(0.8 * data_size), int(0.2 * data_size)])

# sizes = [[256, 0.0001], [128, 0.00005], [64, 0.000025], [32, 0.0000125], [16, 0.00000625], [8, 0.000003125],
#         [4, 0.0000015625]]
sizes = [256, 128, 64, 32, 16, 8, 4]
for batch_size in sizes:
    lr = 5e-05
    torch.random.manual_seed(0)
    writer = SummaryWriter(filename_suffix=f"batch_size_{batch_size},lr_{lr}",
                           comment=f"batch_size_{batch_size},lr_{lr}")
    starttime = time.time()
    testtime = 0

    device = torch.device("mps")

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # train
    network = Network().to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    loss_function = nn.MSELoss().to(device)
    network.load_state_dict(torch.load("blankmodel/blank.pth")[0])
    optimizer.load_state_dict(torch.load("blankmodel/blank.pth")[1])
    optimizer.param_groups[0]['lr'] = lr
    torch.random.manual_seed(0)
    _iter = 0
    for epoch in range(500):
        network.train()
        train_loss = 0
        for inputs, labels in train_loader:
            _iter += 1
            inputs = inputs.to(device).float()
            labels = labels.to(device).float()
            outputs = network(inputs)
            loss = loss_function(outputs, labels)

            train_loss += loss.cpu().item()

            # back propagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            writer.add_scalar("Train Loss / Data", loss.item(), _iter * batch_size)

        train_loss = train_loss / len(train_loader)
        writer.add_scalar("Train Loss / Epoch", train_loss, epoch + 1)
        writer.add_scalar("Epoch / Time", epoch + 1, int(time.time() - starttime))
        test_loss = 0
        with torch.no_grad():
            network.eval()
            for inputs, labels in test_loader:
                inputs = inputs.to(device).float()
                labels = labels.to(device).float()
                outputs = network(inputs)
                loss = loss_function(outputs, labels)

                test_loss += loss.cpu().item()

        test_loss = test_loss / len(test_loader)
        writer.add_scalar("Test Loss / Epoch", test_loss, epoch + 1)
