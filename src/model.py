import torch.nn as nn


class ConvNet(nn.Module):  # 16 -> 12

    def __init__(self, n_classes):
        self.n_classes = n_classes
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.drop = nn.Dropout2d(p=0.6)
        self.bn2 = nn.BatchNorm2d(num_features=12)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(in_features=75 * 75 * 12, out_features=n_classes)

    # Feed forwad function
    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(output)
        output = self.drop(output)
        output = output.view(-1, 12*75*75)
        output = self.fc(output)
        return output
