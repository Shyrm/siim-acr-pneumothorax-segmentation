import pandas as pd
import numpy as np
import cv2
from prepare_probs import CheXDataset

from torch.utils import data
import torchvision

from Architectures import DynamicUnet_Hcolumns
from fastai.vision.learner import cnn_config
from Utils.training_fastai import create_body_local
import torch

from fastai.layers import NormType
from shutil import copyfile


OUTPUT_FOLDER = './Data/PseudoMaskData'

PROB_DATA = './Data/ProbData/Images'
PROB_DATA_TARGETS = './Data/ProbData/Targets.csv'
PROB_DATA_META = './Data/ProbData/Meta.csv'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

STATS = ([0.544, 0.544, 0.544], [0.2645, 0.2645, 0.2645])

PROB_THRESHOLD = 0.15
PIXEL_THRESHOLD = 150


SZ = 256
BATCH_SIZE = 16
NUM_WORKERS = 30

ARCH = torchvision.models.resnet34
STATE_FILE = './FittedModels/UnetResnet34_8430.pth'

if __name__ == '__main__':

    prob_targets = pd.read_csv(PROB_DATA_TARGETS, sep=';', header=0)

    prob_targets = prob_targets[(prob_targets['IsPneumo'] == 1) & (prob_targets['Image'].apply(len) < 40)]


    # dataset = CheXDataset(
    #     img_ids=prob_targets['Image'].values,
    #     xray_path=PROB_DATA,
    #     normalization=STATS
    # )
    #
    # dataloader = data.DataLoader(
    #     dataset=dataset,
    #     batch_size=BATCH_SIZE,
    #     shuffle=False,
    #     num_workers=NUM_WORKERS
    # )
    #
    # # ==================== init model from state dict ==================
    # body = create_body_local(arch=ARCH)
    # model = DynamicUnet_Hcolumns(
    #     body,
    #     n_classes=2,
    #     blur=False,
    #     blur_final=True,
    #     self_attention=False,
    #     y_range=None,
    #     last_cross=True,
    #     bottle=False,
    #     norm_type=NormType
    # )
    #
    # model.load_state_dict(torch.load(STATE_FILE)['model'])
    # model.to(DEVICE)
    # model.eval()
    #
    # # ==================== predict listed images ==================
    # k = 0
    # for img_batch, img_batch_flip in dataloader:
    #
    #     with torch.no_grad():
    #
    #         preds = model(img_batch.to(DEVICE))
    #         preds_flip = model(img_batch_flip.to(DEVICE))
    #
    #         preds = 0.5 * (preds + torch.flip(preds_flip, [-1]))
    #         preds = torch.softmax(preds, dim=1)[:, 1, ...].view(-1, 1, SZ, SZ)
    #
    #         preds = (preds > PROB_THRESHOLD).long()
    #         preds[preds.view(preds.shape[0], -1).sum(-1) < PIXEL_THRESHOLD, ...] = 0
    #
    #         preds = preds.cpu().detach().numpy()
    #         preds = (preds * 255).astype(np.uint8)
    #
    #         n = preds.shape[0]
    #
    #         for i in range(n):
    #
    #             img_name = prob_targets['Image'].values[k + i]
    #             img = np.moveaxis(preds[i], 0, -1)
    #
    #             cv2.imwrite(f'{OUTPUT_FOLDER}/mask/{img_name}', img)
    #             copyfile(f'{PROB_DATA}/{img_name}', f'{OUTPUT_FOLDER}/train/{img_name}')
    #
    #         k += n

    # ==================== update metadata ==============================
    prob_meta = pd.read_csv(PROB_DATA_META, sep=';', header=0)
    prob_meta = prob_meta[prob_meta['ImageId'].isin(prob_targets['Image'])]
    prob_meta['ImageId'] = prob_meta['ImageId'].apply(lambda x: x[:-4])

    original_meta = pd.read_csv(f'{OUTPUT_FOLDER}/train_meta.csv', sep=';', header=0)
    original_meta = original_meta[prob_meta.columns]

    meta = pd.concat([prob_meta, original_meta])

    original_targets = pd.read_csv(f'{OUTPUT_FOLDER}/Targets.csv', sep=';', header=0)
    targets = pd.concat([prob_targets, original_targets])

    meta.to_csv(f'{OUTPUT_FOLDER}/train_meta.csv', sep=';', header=True, index=False)
    targets.to_csv(f'{OUTPUT_FOLDER}/Targets.csv', sep=';', header=True, index=False)