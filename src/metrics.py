import matplotlib.pyplot as plt


def plot_acc(train_acc, val_acc):
    """[summary]

    Args:
        train_acc (array): training accuracy data
        val_acc (array): validation accuracy data
    """
    plt.title("Train-Validation Accuracy")
    plt.plot(train_acc, label='train')
    plt.plot(val_acc, label='validation')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('accuracy', fontsize=12)
    plt.legend(loc='best')
    plt.show()
