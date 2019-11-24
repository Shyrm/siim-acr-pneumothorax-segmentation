from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from itertools import filterfalse
import numpy as np
from fastai.layers import FlattenedLoss



def dice_loss_glob(smooth=1.):

    def dice_loss(pred, target):

        n = target.shape[0]

        pred = torch.softmax(pred, dim=1)[:, 1, ...].view(n, -1)
        target = target.view(n, -1).float()

        intersect = (pred * target).sum(-1).float()
        union = (pred + target).sum(-1).float()

        dc = (2.0 * intersect + smooth) / (union + smooth)

        return (torch.ones_like(dc) - dc).mean()

    return dice_loss


class ProbDiceFlat:

    def __init__(self, *args, axis: int = -1, dice_share: float = 0.5, **kwargs):

        self.pl = FlattenedLoss(nn.CrossEntropyLoss, *args, axis=axis, **kwargs)
        self.dl = dice_loss_glob(smooth=1.)

        self.dice_share = dice_share

    def __call__(self, input, target, **kwargs):

        return (1 - self.dice_share) * self.pl(input, target) + self.dice_share * self.dl(input, target)


class WeightedBCELoss(nn.Module):

    def __init__(self, positive_factor=1., reduction='mean'):

        super(WeightedBCELoss, self).__init__()
        self.positive_factor = positive_factor
        self.reduction = reduction
        self.update_on_training = False
        self.update_on_epoch = None

    def forward(self, output, target):

        output = output.view(output.size(0), -1)
        target = target.view(target.size(0), -1)

        w = self.positive_factor * target + torch.ones_like(target) - target
        prob_loss = nn.BCELoss(weight=w, reduction=self.reduction)

        return prob_loss(output, target)


class MFELoss(nn.Module):

    def __init__(self, positive_factor=1., reduction='mean'):

        super(MFELoss, self).__init__()
        self.positive_factor = positive_factor
        self.reduction = reduction
        self.update_on_training = False
        self.update_on_epoch = None

    def forward(self, output, target):

        output = output.view(output.size(0), -1)
        target = target.view(target.size(0), -1)

        fp = self.positive_factor * (output * (torch.ones_like(target) - target)).mean()
        fn = ((1 - output) * target).mean()

        return fp + fn


class ProbDiceLoss(nn.Module):

    def __init__(self, bce_weight=0.5, dice_weight=0.5, prob_positive_factor=1.,
                 dice_smooth=1., dice_prob_thr=None, dice_noise_thr=None,
                 reduction='mean', factor=1, update_on_training=False,
                 update_on_epoch=None, update_factor=2, update_side='dice'):

        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.prob_positive_factor = prob_positive_factor
        self.reduction = reduction
        self.dice_smooth = dice_smooth
        self.dice_prob_thr = dice_prob_thr
        self.dice_noise_thr = dice_noise_thr
        self.factor = factor
        self.update_on_training = update_on_training
        self.update_factor = update_factor
        self.update_on_epoch = update_on_epoch
        self.update_side = update_side
        self.prob_loss = WeightedBCELoss(positive_factor=prob_positive_factor, reduction=reduction)
        self.dice_loss = SoftDiceLoss(smooth=dice_smooth, prob_thr=dice_prob_thr, noise_thr=dice_noise_thr)

    def forward(self, output, target):

        target = target.to(torch.float32)

        bce_loss = self.prob_loss(output, target) * self.bce_weight

        dice_loss = self.dice_loss(output, target) * self.dice_weight

        return self.factor * (bce_loss + dice_loss)

    def update(self):

        if self.update_side == 'dice':
            self.bce_weight /= self.update_factor
            self.dice_weight = 1. - self.bce_weight
        else:
            self.dice_weight /= self.update_factor
            self.bce_weight = 1. - self.dice_weight


class FullLogLoss(nn.Module):

    def __init__(self, bce_weight=0.5, dice_weight=0.5, reduction='mean', factor=1):

        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.reduction = reduction
        self.factor = factor
        self.prob_loss = nn.BCELoss(reduction=reduction)
        self.dice_loss = SoftDiceLoss(smooth=1.)

    def forward(self, output, target):

        target = target.to(torch.float32)

        bce_loss = self.prob_loss(output, target) * self.bce_weight

        dice_loss = self.dice_loss(output, target) * self.dice_weight

        return self.factor * (bce_loss + dice_loss)


def dice_coeff(pred, target, smooth=1., prob_thr=None, noise_thr=None):

    if prob_thr is not None and noise_thr is not None:
        pred = (pred > prob_thr).long()
        pred[pred.sum(-1) < noise_thr, ...] = 0.0

    num = pred.size(0)
    m1 = pred.view(num, -1).float()
    m2 = target.view(num, -1).float()

    intersection = (m1 * m2).sum(dim=1).float()
    cardinality = m1.sum(dim=1) + m2.sum(dim=1)

    return (2. * intersection + smooth) / (cardinality + smooth)


class SoftDiceLoss(nn.Module):

    def __init__(self, smooth=1., prob_thr=None, noise_thr=None):

        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.prob_thr = prob_thr
        self.noise_thr = noise_thr

    def forward(self, output, target):

        score = dice_coeff(output, target, smooth=self.smooth, prob_thr=self.prob_thr, noise_thr=self.noise_thr)
        score = 1 - score.mean()

        return score


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.elu(errors_sorted) + 1, Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = filterfalse(np.isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


class LovaszProbLoss(nn.Module):
    def __init__(self, lovasz_weight=0.5, prob_weight=0.5, min_mask_pixels=0):
        super().__init__()
        self.lovasz_weight = lovasz_weight
        self.prob_weight = prob_weight
        self.min_mask_pixels = min_mask_pixels
        self.prob_loss = nn.BCELoss()

    def forward(self, output, target):
        segm, prob_pred = output

        prob_trg = target.view(target.size(0), -1).sum(dim=1) > self.min_mask_pixels
        prob_trg = prob_trg.to(torch.float32)
        if self.prob_weight > 0:
            prob = self.prob_loss(prob_pred, prob_trg) * self.prob_weight
        else:
            prob = 0

        if self.lovasz_weight > 0:
            lovasz = lovasz_hinge(segm.squeeze(1), target.squeeze(1)) \
                     * self.lovasz_weight
        else:
            lovasz = 0

        return prob + lovasz


class LovaszLoss:

    def __call__(self, input, target):

        input = input[:, 1, ...]

        lovasz = lovasz_hinge(input.squeeze(1), target.squeeze(1))

        return lovasz
