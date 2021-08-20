import matplotlib.pyplot as plt
from loader import load_data
from train import train


def plot_acc(train_acc, val_acc):
    plt.title("Train-Validation Accuracy")
    plt.plot(train_acc, label='train')
    plt.plot(val_acc, label='validation')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('accuracy', fontsize=12)
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    batch_size = 256
    test_size = 0.33
    train_dataloader, test_dataloader = load_data(batch_size, test_size)
    train_acc, val_acc = train(train_dataloader, test_dataloader)
    plot_acc(train_acc, val_acc)
