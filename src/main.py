import torch
from sklearn.model_selection import StratifiedKFold
from model import CNN, train_model
from loader import load_config, load_data
from evaluate import validate_model, plot_metrics


def main():
    n_epoch, device, criterion, optimizer, model = load_config(CNN())
    x, y = load_data(batch_size=64)
    kfold = StratifiedKFold(n_splits=10)
    train_losses = []
    valid_losses = []

    for train_index, test_index in kfold.split(x, y):
        train_loss = 0.0
        valid_loss = 0.0

        # TRAIN-TEST SPLIT
        x_train, y_train = x[train_index], y[train_index]
        x_test, y_test = x[test_index], y[test_index]

        # MODEL TRAINING
        train_loss = train_model(model, x_train, y_train,
                                 device, optimizer, criterion, n_epoch)

        # MODEL EVALUATION
        valid_loss = validate_model(model, x_test, y_test,
                                    device, criterion, n_epoch)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

    plot_metrics(train_losses, valid_losses)
    torch.save(model.state_dict(), 'best_checkpoint.pt')


if __name__ == '__main__':
    main()
