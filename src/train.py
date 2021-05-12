import torch
import torch.nn as nn
from timeit import default_timer as timer
import pandas as pd
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(model, train_ldr, test_ldr):
    if torch.cuda.is_available():
        print(f'Training on {torch.cuda.get_device_name()}.\n')
    else:
        print('Training on CPU.\n')
    print(f'Training on {len(train_ldr.dataset)} samples.')
    print(f'Validation on {len(test_ldr.dataset)} samples.')
    print(f'Number of classes: {model.n_classes}\n')
    model = model.to(device)
    history = []

    ''' OPTIMIZATION FUNCTION '''
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    ''' LOSS FUNCTION '''
    criterion = nn.CrossEntropyLoss()

    overall_start = timer()
    n_epochs = 25
    for epoch in range(n_epochs):
        # Initialization
        train_loss = 0.0
        val_loss = 0.0
        train_acc = 0.0
        val_acc = 0.0
        model.train()  # Set model parameters to be learnable
        # Training loop
        for i, (data, target) in enumerate(train_ldr):
            data, target = data.to(device, dtype=torch.float), \
                target.to(device, dtype=torch.long)
            # Forward pass
            optimizer.zero_grad()
            output = model(data).view(-1, model.n_classes)
            # Backward pass
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            # Tracking
            train_loss += loss.item() * data.size(0)
            _, pred = torch.max(output, dim=1)
            correct = pred.eq(target.data.view_as(pred))
            accuracy = torch.mean(correct.type(torch.FloatTensor))
            train_acc += accuracy.item() * data.size(0)
            print(f'Epoch: {epoch + 1}\t'
                  f'{100 * (i + 1) / len(train_ldr):.2f}%\n',
                  end='\r')
        # End-of-epoch validation
        else:
            with torch.no_grad():
                model.eval()
                # Validation loop
                for data, target in test_ldr:
                    data, target = data.to(device, dtype=torch.float), \
                        target.to(device, dtype=torch.long)
                    output = model(data).view(-1, model.n_classes)
                    loss = criterion(output, target)
                    val_loss += loss.item() * data.size(0)
                    _, pred = torch.max(output, dim=1)
                    correct = pred.eq(target.data.view_as(pred))
                    accuracy = torch.mean(correct.type(torch.FloatTensor))
                    val_acc += accuracy.item() * data.size(0)
                # Average loss and accuracy
                train_loss = train_loss / len(train_ldr.dataset)
                val_loss = val_loss / len(test_ldr.dataset)
                train_acc = train_acc / len(train_ldr.dataset)
                val_acc = val_acc / len(test_ldr.dataset)
                history.append([train_loss, val_loss,
                                train_acc, val_acc])
                print(f'\n\t\tTraining loss: {train_loss:.4f} \t\t'
                      f'Validation loss: {val_loss:.4f}')
                print(f'\t\tTraining accuracy: {100 * train_acc:.2f}%\t'
                      f'Validation accuracy: {100 * val_acc:.2f}%\n')
    total_time = timer() - overall_start
    print(f'\n{total_time:.2f} total seconds elapsed. Average of '
          f'{total_time / n_epochs:.2f} seconds per epoch.')
    history = pd.DataFrame(history,
                           columns=['train_loss', 'val_loss',
                                    'train_acc', 'val_acc'])
    return model, history


def plot(history):
    # Loss plot
    plt.figure()
    for c in ['train_loss', 'val_loss']:
        plt.plot(history[c], label=c)
    plt.grid()
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average cross-entropy loss')
    plt.title('Training and validation loss')
    # Accuracy plot
    plt.figure()
    for c in ['train_acc', 'val_acc']:
        plt.plot(history[c] * 100, label=c)
    plt.grid()
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average accuracy')
    plt.title('Training and validation accuracy')
    plt.show()
