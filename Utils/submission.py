import numpy as np
import PIL
from tqdm import tqdm
from Utils.mask_functions import mask2rle


OUT_SIZE = 1024


def prepare_submission(preds, prob_threshold=0.5, pixel_threshold=20):

    preds[preds.view(preds.shape[0], -1).sum(-1) < pixel_threshold, ...] = 0.0
    preds = (preds > prob_threshold).long()

    rles = []
    for p in tqdm(preds):
        p = p.cpu().detach().numpy()
        im = PIL.Image.fromarray((p.T * 255).astype(np.uint8)).resize((1024, 1024))
        im = np.asarray(im)
        rles.append(mask2rle(im, 1024, 1024))

    return rles
