import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score

from Utils.preprocessing import Preprocessing

import torch
from torch import nn
import torchvision

from Utils.training_fastai import create_cnn_model
from torchvision import transforms
from torch.utils import data
import cv2
from PIL import Image
from sklearn.metrics import accuracy_score


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NORM = 255.

FLIP = True

# ARCH = pretrainedmodels.se_resnext50_32x4d()
# STATE_FILE = './FittedModels/ProbModel_v02/best_model.pth'

ARCH = torchvision.models.resnet34
STATE_FILE = './FittedModels/PostCheXRayResNet34.pth'

NUM_CLASSES = 2
BATCH_SIZE = 16
NUM_WORKERS = 30
STATS = ([0.544, 0.544, 0.544], [0.2645, 0.2645, 0.2645])

FOLDERS = [
    {
        'folder': './Data/MaskData/test',
        'idx': './Data/MaskData/test_meta.csv',
        'add_png': True,
        'target_column': None,
        'predictions_output': './Data/MaskData/TestPneumoProb.csv'
    },
    {
        'folder': './Data/ProbData/Images',
        'idx': './Data/ProbData/Targets.csv',
        'add_png': False,
        'target_column': 'IsPneumo',
        'predictions_output': './Data/ProbData/CheXRayPneumoProb.csv'
    }
]


class CheXDataset(data.Dataset):

    @staticmethod
    def grey2rgb(image):
        return np.concatenate([image[np.newaxis, :] for _ in range(3)], axis=0)

    def __init__(self,
                 img_ids,
                 xray_path,
                 norm=NORM,
                 normalization=STATS):

        self.img_ids = img_ids
        self.xray_path = xray_path
        self.norm = norm
        self.normalization = normalization

    def __getitem__(self, idx):

        img_path = os.path.join(self.xray_path, self.img_ids[idx])

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_flip = np.array(transforms.functional.hflip(Image.fromarray(img)))

        img = CheXDataset.grey2rgb(img) / self.norm
        img_flip = CheXDataset.grey2rgb(img_flip) / self.norm

        img = torch.as_tensor(img, dtype=torch.float32)
        img_flip = torch.as_tensor(img_flip, dtype=torch.float32)

        if self.normalization is not None:
            img = transforms.Normalize(self.normalization[0], self.normalization[1])(img)
            img_flip = transforms.Normalize(self.normalization[0], self.normalization[1])(img_flip)

        return img, img_flip

    def __len__(self):

        return len(self.img_ids)


if __name__ == '__main__':

    model = create_cnn_model(
        base_arch=ARCH,
        nc=NUM_CLASSES
    )

    model.load_state_dict(torch.load(STATE_FILE)['model'])
    model.to(DEVICE)
    model.eval()

    for data_dict in FOLDERS:

        # ========================= read image indexes and targets if available ==========================
        idx = pd.read_csv(data_dict['idx'], sep=';', header=0)

        if data_dict['target_column'] is not None:
            targets = idx[data_dict['target_column']].values
        else:
            targets = None

        idx = list(idx[idx.columns[0]].values)
        if data_dict['add_png']:
            idx = [f'{i}.png' for i in idx]

        # ========================= get predictions from model ==========================================
        dataset = CheXDataset(img_ids=idx, xray_path=data_dict['folder'])
        dataloader = data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        predictions = np.zeros(len(idx))
        k = 0
        for img_batch, img_batch_flip in dataloader:

            with torch.no_grad():

                preds = model(img_batch.to(DEVICE))

                if FLIP:
                    preds = 0.5 * (preds + model(img_batch_flip.to(DEVICE)))

                preds = torch.softmax(preds, dim=1)[:, 1].view(-1, 1)
                preds = preds.cpu().detach().numpy()
                predictions[k:k+preds.shape[0]] = preds.flatten()

                k += preds.shape[0]

                if k % (BATCH_SIZE * 200) == 0:
                    print(k)

        # ========================== find optimal threshold if target available =============================
        if targets is not None:

            for thr in np.linspace(0, 1, 20):

                acc = accuracy_score(
                    y_true=targets,
                    y_pred=np.where(predictions >= thr, 1, 0)
                )

                print(f'Threshold: {thr}, accuracy: {acc}')

        # ========================== store results into file ==============================================
        res = pd.DataFrame(
            data={
                'ImageId': idx,
                'PneumoProb': predictions
            }, columns=['ImageId', 'PneumoProb']
        )

        res.to_csv(data_dict['predictions_output'], sep=';', header=True, index=False)