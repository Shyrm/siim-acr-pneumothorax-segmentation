from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from itertools import filterfalse
import numpy as np
from sklearn.metrics import roc_auc_score
from fastai.metrics import accuracy_thresh


def dice_glob(smooth=1., prob_thr=None, noise_thr=None):

    def dice(pred, target):

        n = target.shape[0]
        pred = torch.softmax(pred, dim=1)[:, 1, ...].view(n, -1)
        pred = (pred > prob_thr).long()
        pred[pred.sum(-1) < noise_thr, ...] = 0.0
        # input = input.argmax(dim=1).view(n,-1)
        target = target.view(n, -1)
        intersect = (pred * target).sum(-1).float()
        union = (pred + target).sum(-1).float()

        return ((2.0 * intersect + smooth) / (union + smooth)).mean()

    return dice


def acc(threshold=0.5):

    def accuracy(pred, target):

        n = target.shape[0]
        pred = torch.softmax(pred, dim=1)[:, 1, ...].view(n, -1)
        # pred = (pred > threshold).long()
        pred = (pred > threshold).float()

        target = target.view(n, -1)

        return accuracy_thresh(y_pred=pred, y_true=target, thresh=threshold)

    return accuracy

# def dice_glob(smooth=1., prob_thr=None, noise_thr=None):
#
#     def dice(pred, target):
#
#         n = target.shape[0]
#
#         pred = (pred > prob_thr).float()
#         pred[pred.sum(-1) < noise_thr, ...] = 0.0
#
#         pred = pred.view(n, -1)
#         target = target.view(n, -1)
#
#         intersect = (pred * target).sum(-1).float()
#         union = (pred + target).sum(-1).float()
#
#         return ((2.0 * intersect + smooth) / (union + smooth)).mean()
#
#     return dice


class Accuracy(nn.Module):

    def __init__(self, threshold=0.5):

        super(Accuracy, self).__init__()
        self.threshold = threshold

    def forward(self, output, target):

        output = (output > self.threshold).float()
        target = target.float()

        score = (output == target).float()
        score = score.mean()

        return score


class AUC(nn.Module):

    def __init__(self):

        super(AUC, self).__init__()

    def forward(self, output, target):

        y_true = target.cpu().detach().numpy()
        y_pred = output.cpu().detach().numpy()

        return roc_auc_score(y_true, y_pred)
