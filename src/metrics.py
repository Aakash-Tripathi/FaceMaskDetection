import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score


def plot_acc(train_acc, val_acc):
    plt.title("Train-Validation Accuracy")
    plt.plot(train_acc, label='train')
    plt.plot(val_acc, label='validation')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('accuracy', fontsize=12)
    plt.legend(loc='best')
    plt.show()


def make_metrics(y_true, y_pred):
    accuracy = []
    precision = []
    sensitivity = []
    specificity = []
    roc_auc = []

    for i in range(len(y_pred)):
        tn, fp, fn, tp = confusion_matrix(y_true[i], y_pred[i]).ravel()
        cm = confusion_matrix(y_true[i], y_pred[i])
        accuracy.append((tp + tn) / (tp + tn + fp + fn))
        precision.append(tp / (tp + fp))
        sensitivity.append(tp / (tp + fn))
        specificity.append(tn / (tn + fp))
        roc_auc.append(roc_auc_score(y_true[i], y_pred[i]))
        print(tp, tn, fp, fn)

    acc_mean = np.mean(accuracy)
    prec_mean = np.mean(precision)
    sens_mean = np.mean(sensitivity)
    spec_mean = np.mean(specificity)
    roc_mean = np.mean(roc_auc)

    print(accuracy, acc_mean)
    print(precision, prec_mean)
    print(sensitivity, sens_mean)
    print(specificity, spec_mean)
    print(roc_auc, roc_mean)

    return cm


'''
y_true = [[1, 0, 0, 1, 1], [1, 0, 0, 1, 1]]
y_pred = [[1, 0, 0, 0, 1], [1, 0, 0, 1, 1]]
cm = make_metrics(y_true, y_pred)

print("\nPlatform Info \n-------------\n", platform.uname())

df_cm = pd.DataFrame(cm, range(2), range(2))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
plt.show()
'''
