import matplotlib.pyplot as plt
import torch

import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import balanced_accuracy_score


def validate_model(model, x, y, device, criterion, n_epoch):
    model.eval()
    valid_loss = 0.0
    for i in range(n_epoch):
        data = x.to(device)
        target = y.to(device)
        output = model(data)
        loss = criterion(output, target)
        valid_loss += loss.item()*data.size(0)
    valid_loss = valid_loss/len(y)
    return valid_loss


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


def make_metrics(logits_all, labels_all):
    accuracy = []
    precision = []
    sensitivity = []
    specificity = []
    roc_auc = []
    prc_auc = []
    balanced_acc = []
    for i in range(len(logits_all)):
        tn, fp, fn, tp = confusion_matrix(
            labels_all[i], np.round(logits_all[i])).ravel()
        accuracy.append((tp + tn) / (tp + tn + fp + fn))
        precision.append(tp / (tp + fp))
        sensitivity.append(tp / (tp + fn))
        specificity.append(tn / (tn + fp))
        roc_auc.append(roc_auc_score(labels_all[i], logits_all[i]))
        prc_auc.append(average_precision_score(labels_all[i], logits_all[i]))
        balanced_acc.append(balanced_accuracy_score(
            labels_all[i], np.round(logits_all[i])))
    acc_mean, acc_confidence_interval = mean_confidence_interval(accuracy)
    print('Accuracy Mean and confidence interval: {:4f}, {:4f}'.format(
        acc_mean, acc_confidence_interval))

    prec_mean, prec_confidence_interval = mean_confidence_interval(precision)
    print('Precision Mean and confidence interval: {:4f}, {:4f}'.format(
        prec_mean, prec_confidence_interval))

    sens_mean, sens_confidence_interval = mean_confidence_interval(sensitivity)
    print('Sensitivity Mean and confidence interval: {:4f}, {:4f}'.format(
        sens_mean, sens_confidence_interval))

    spec_mean, spec_confidence_interval = mean_confidence_interval(specificity)
    print('Specificity Mean and confidence interval: {:4f}, {:4f}'.format(
        spec_mean, spec_confidence_interval))

    roc_mean, roc_confidence_interval = mean_confidence_interval(roc_auc)
    print('ROC_AUC Mean and confidence interval: {:4f}, {:4f}'.format(
        roc_mean, roc_confidence_interval))

    prc_mean, prc_confidence_interval = mean_confidence_interval(prc_auc)
    print('PRC_AUC Mean and confidence interval: {:4f}, {:4f}'.format(
        prc_mean, prc_confidence_interval))

    bacc_mean, bacc_confidence_interval = mean_confidence_interval(
        balanced_acc)
    print('Balanced Accuracy Mean and confidence interval: {:4f}, {:4f}'.format(
        bacc_mean, bacc_confidence_interval))


def plot_metrics(train_losses, valid_losses):
    plt.plot(train_losses, label='Training loss')
    plt.plot(valid_losses, label='Validation loss')
    plt.xlabel("K-Fold")
    plt.ylabel("Loss")
    plt.legend(frameon=False)
    plt.show()
