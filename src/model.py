import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=48,
                               kernel_size=1, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=24,
                               kernel_size=1, stride=1, padding=1)
        self.drop = nn.Dropout2d(p=0.5)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(in_features=154*154*24, out_features=2)

    def forward(self, input):
        output = self.conv1(input)
        output = self.relu1(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.drop(output)
        # print(output.shape)
        output = output.view(output.shape[0], -1)
        # print(output.shape)
        output = self.fc(output)
        return output


def train_model(model, x, y, device, optimizer, criterion, n_epoch):
    model.train()
    train_loss = 0.0
    for i in range(n_epoch):
        data = x.to(device)
        target = y.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
    train_loss = train_loss/len(y)
    return train_loss
