import numpy as np
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm


MIN_PIXEL_THRESHOLDS = np.arange(0, 1000, 50)
PROBABILITY_THRESHOLDS = np.arange(0.0, 1.0, 0.05)


def dice_overall(preds, targs):
    n = preds.shape[0]
    preds = preds.view(n, -1)
    targs = targs.view(n, -1)
    intersect = (preds * targs).sum(-1).float()
    union = (preds+targs).sum(-1).float()
    u0 = union==0
    intersect[u0] = 1
    union[u0] = 2
    return (2. * intersect / union).mean()


def fit_optimal_thresholds(preds, acts, prob_thr=PROBABILITY_THRESHOLDS, pixel_thr=MIN_PIXEL_THRESHOLDS):

    grid = []
    for pitrh in tqdm(pixel_thr):
        for prtrh in tqdm(prob_thr):
            p = preds
            p[p.view(p.shape[0], -1).sum(-1) < pitrh, ...] = 0.0
            p = (p > prtrh).long()
            grid.append(OrderedDict({'pitrh': pitrh, 'prtrh': prtrh, 'dice': dice_overall(p, acts).item()}))

    grid = pd.DataFrame.from_records(grid)
    best_thresholds = grid.sort_values(by='dice', ascending=False).iloc[0]

    return best_thresholds['dice'], best_thresholds['pitrh'], best_thresholds['prtrh']
