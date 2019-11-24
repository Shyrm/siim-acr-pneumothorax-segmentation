import torch
import gc
from fastai.basic_data import DatasetType
from fastai.vision import flip_lr
import numpy as np


def pred_with_flip(learn, ds_type=DatasetType.Valid, probs=None):

    # get prediction
    preds, acts = learn.get_preds(ds_type)
    preds = preds[:, 1, ...]

    # add flip to dataset and get prediction
    learn.data.dl(ds_type).dl.dataset.tfms.append(flip_lr())
    preds_lr, acts = learn.get_preds(ds_type)
    del learn.data.dl(ds_type).dl.dataset.tfms[-1]
    preds_lr = preds_lr[:, 1, ...]

    acts = acts.squeeze()
    preds = 0.5 * (preds + torch.flip(preds_lr, [-1]))

    del preds_lr
    gc.collect()
    torch.cuda.empty_cache()

    if probs is not None:
        for i in range(len(probs)):
            p = torch.tensor(probs[i], dtype=torch.float).unsqueeze(1).unsqueeze(2)
            preds[i] = preds[i] * p

    return preds, acts


def pred_tta(learn, ds_type=DatasetType.Valid):

    # get prediction
    preds, acts = learn.TTA(ds_type=ds_type)
    preds = preds[:, 1, ...]

    return preds, acts


def predict_prob(model, dataloader, device='cpu'):
    model.eval()
    model.to(device)

    # PREDICT MASKS FOR VALIDATION
    n = len(dataloader.dataset)
    probs = np.zeros(shape=(n,))
    k = 0
    for inputs in dataloader:
        with torch.no_grad():
            prob = model(inputs.to(device))
            probs[k:(k + inputs.shape[0])] = prob.cpu().detach().numpy()
        k = k + inputs.shape[0]

    return probs
