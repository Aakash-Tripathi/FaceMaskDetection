import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import models
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net = models.resnet18(pretrained=True)
net = net.cuda() if device else net

num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 128)
net.fc = net.fc.cuda()

n_epochs = 50
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)


def train(train_dataloader, test_dataloader):

    valid_loss_min = np.Inf
    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []

    for epoch in range(1, n_epochs+1):

        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (data_, target_) in enumerate(train_dataloader):
            data_, target_ = data_.to(device), target_.to(device)
            optimizer.zero_grad()
            outputs = net(data_)
            loss = criterion(outputs, target_)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, pred = torch.max(outputs, dim=1)
            correct += torch.sum(pred == target_).item()
            total += target_.size(0)
            if (batch_idx) % 20 == 0:
                print(f'\nEpoch [{epoch}/{n_epochs}], Loss: {loss.item():.4f}')
        train_acc.append(100*correct/total)
        train_loss.append(running_loss/len(train_dataloader))
        train_loss_val = np.mean(train_loss)
        mean_acc = 100*correct/total
        print(f"train-loss: {train_loss_val:.4f}, train-acc: {mean_acc:.4f}")

        batch_loss = 0
        total_t = 0
        correct_t = 0
        with torch.no_grad():
            net.eval()
            for data_t, target_t in (test_dataloader):
                data_t, target_t = data_t.to(device), target_t.to(device)
                outputs_t = net(data_t)
                loss_t = criterion(outputs_t, target_t)
                batch_loss += loss_t.item()
                _, pred_t = torch.max(outputs_t, dim=1)
                correct_t += torch.sum(pred_t == target_t).item()
                total_t += target_t.size(0)
            val_acc.append(100 * correct_t/total_t)
            val_loss.append(batch_loss/len(test_dataloader))
            network_learned = batch_loss < valid_loss_min
            valloss = np.mean(val_loss)
            valacc = 100*correct_t/total_t
            print(f'val loss: {valloss:.4f}, val acc: {valacc:.4f}\n')
            if network_learned:
                valid_loss_min = batch_loss
                torch.save(net.state_dict(), os.getcwd()+'/models/resnet.pt')

        net.train()

    return train_acc, val_acc
