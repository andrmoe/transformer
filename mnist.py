import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import optim
from mnist_data import make_training_data_loader, make_test_data_loader, manual_training_dataloader


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def activations(self, x):
        xs = []
        xs.append(F.relu(F.max_pool2d(self.conv1(x), 2)))
        xs.append(F.relu(F.max_pool2d(self.conv2_drop(self.conv2(xs[-1])), 2)))
        xs.append(xs[-1].view(-1, 320))
        xs.append(F.relu(self.fc1(xs[-1])))
        xs.append(F.dropout(xs[-1], training=self.training))
        xs.append(F.relu(self.fc2(xs[-1])))
        xs.append(F.softmax(xs[-1], dim=1))
        return xs

    def forward(self, x):
        return self.activations(x)[-1]


def train(epoch, loaders, model, optimiser, loss_fn, device):
    model.train()
    for batch_index, (data, target) in enumerate(loaders["train"]):
        data, target = data.to(device), target.to(device)
        optimiser.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimiser.step()
        if batch_index % 100 == 0:
            print(f"Train epoch: {epoch}, [{batch_index*len(data)}/{len(loaders["train"].dataset)}]")


def test(test_data_loader, model, loss_fn, device):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_data_loader.dataset)
    print(f"Average loss; {test_loss}")
    print(f"Accuracy: {correct/len(test_data_loader.dataset)*100}%")


def main():
    loaders = {"train": manual_training_dataloader(),
               "test":  make_test_data_loader()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    optimiser = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, 11):
        train(epoch, loaders, model, optimiser, loss_fn, device)
        test(loaders["test"], model, loss_fn, device)

    torch.save(model.state_dict(), "model1.pth")


if __name__ == "__main__":
    main()
