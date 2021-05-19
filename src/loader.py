from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import os
import pandas as pd
import matplotlib.image as img


class FMDDataset(Dataset):
    def __init__(self, data, path, transform=None):
        super().__init__()
        self.data = data.values
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name, label = self.data[index]
        img_path = os.path.join(self.path, img_name)
        image = img.imread(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


path = 'D:/Projects/FaceMaskDetection/data/'
labels = pd.read_csv(path+'train.csv')
train_path = path+'FMD/'


def load_data(batch_size):
    train_transform = transforms.Compose([transforms.ToPILImage(),
                                          transforms.ToTensor(),
                                          transforms.Resize((150, 150)),
                                          transforms.Normalize(0, 1)])
    data = FMDDataset(labels, train_path, train_transform)
    data_loader = DataLoader(dataset=data,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=2)
    for data, target in data_loader:
        data = data
        target = target
    return data, target


def load_config(model):
    n_epoch = 5
    learning_rate = 0.001
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model = model.to(device)
    return n_epoch, device, criterion, optimizer, model
