import fastai
from fastai.vision import *
from Utils.mask_functions import *
from fastai.callbacks import CSVLogger, EarlyStoppingCallback, SaveModelCallback
import gc
from sklearn.model_selection import KFold

import pretrainedmodels

from Architectures import DynamicUnet_Hcolumns
from Architectures.se_resnext_unet import SeResNeXt50Hyper
from Architectures.resnet_unet import ResNet34Hyper
from Architectures.fast_ai_unet_attention import DynamicUnetAttention_Hcolumns

from Utils.utils import seed_everything
from Utils.losses import ProbDiceFlat, LovaszLoss
from Utils.metrics import dice_glob
from Utils.utils import AccumulateStep, set_BN_momentum
from Utils.training_fastai import unet_learner_e
from Utils.generators_fastai import get_data_fastai
from Utils.predict_functions import pred_with_flip, pred_tta
from Utils.threshold_tuner import fit_optimal_thresholds
from Utils.submission import prepare_submission
from Utils.preprocessing import Preprocessing


import torchvision


TRAIN_MODE = True


SZ = 1024
BATCH_SIZE = 16
n_acc = 64 // BATCH_SIZE  # gradinet accumulation steps
nfolds = 4
SEED = 2019
seed_everything(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


NOISE_THRESHOLD = 300
PROB_THRESHOLD = 0.25

STATS = ([0.525, 0.525, 0.525], [0.2546, 0.2546, 0.2546])
DATA_PATH = './Data'

TRAIN = f'{DATA_PATH}/train'
TEST = f'{DATA_PATH}/test'
MASKS = f'{DATA_PATH}/mask'

TRAIN_RLE_FILE = f'{DATA_PATH}/train_rle.csv'
TRAIN_XRAY_PATH = f'{DATA_PATH}/train_small_xray/'
TRAIN_MASK_PATH = f'{DATA_PATH}/train_small_mask/'
TEST_XRAY_PATH = f'{DATA_PATH}/test_small_xray/'

SAMPLE_SUBMISSION_FILE = f'{DATA_PATH}/sample_submission.csv'

TRAIN_IDX = f'{DATA_PATH}/TrainIdx.csv'
VALID_IDX = f'{DATA_PATH}/ValidIdx.csv'
TEST_IDX = f'{DATA_PATH}/TestIdx.csv'

VALID_PROBS = f'{DATA_PATH}/CheXRayPneumoProb.csv'
TEST_PROBS = f'{DATA_PATH}/TestPneumoProb.csv'


MODEL = DynamicUnet_Hcolumns
# MODEL = DynamicUnetAttention_Hcolumns
# MODEL = SeResNeXt50Hyper
# MODEL = ResNet34Hyper

PRETRAINED = './FittedModels/CheXNetResNet34.pth'
IS_SE_RESNEXT = True if MODEL == SeResNeXt50Hyper else False
ENCODER = torchvision.models.resnet34
# ENCODER = torchvision.models.densenet121
# ENCODER = torchvision.models.resnet50
# ENCODER = pretrainedmodels.resnet34
# ENCODER = pretrainedmodels.se_resnext50_32x4d

NUM_CLASSES = 2
Y_RANGE = None
BLUR = False
SELF_ATTENTION = False
BOTTLE = False

NUM_EPOCHS = 250
PATIENCE = 10
NUM_WORKERS = 30

MODELS_FOLDER = './FittedModels'
LOGGING_FOLDER = f'{MODELS_FOLDER}/Model_v01'
SUBMISSIONS_FOLDER = './Submissions'
SUBMISSION_FILE = 'Submission_v004.csv'


HS_MODEL = None
# HS_MODEL = f'{MODELS_FOLDER}/Model_v02/best_model.pth'


CRITERION = CrossEntropyFlat(axis=1)
CRITERION_ADD = ProbDiceFlat(axis=1, dice_share=0.5)
# CRITERION = LovaszLoss()

METRIC = dice_glob(
    smooth=1.,
    prob_thr=PROB_THRESHOLD,
    noise_thr=NOISE_THRESHOLD
)


if __name__ == '__main__':

    # create folders if not exists
    if not os.path.exists(MODELS_FOLDER):
        os.makedirs(MODELS_FOLDER)

    if not os.path.exists(SUBMISSIONS_FOLDER):
        os.makedirs(SUBMISSIONS_FOLDER)

    if not os.path.exists(LOGGING_FOLDER):
        os.makedirs(LOGGING_FOLDER)

    # load train/validation/test indexes
    # train_idx = pd.read_csv(TRAIN_IDX, sep=';', header=0)['Idx'].values
    valid_idx = pd.read_csv(VALID_IDX, sep=';', header=0)['Idx'].values
    # test_idx = pd.read_csv(TEST_IDX, sep=';', header=0)['Idx'].values

    data = get_data_fastai(
        train_path=TRAIN,
        valid_idx=valid_idx,
        test_path=TEST,
        img_size=(SZ, SZ),
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        normalization=STATS
    )

    # ============================== model learning ==============================
    # init model learner
    learn = unet_learner_e(
        data=data,
        arch=ENCODER,
        unet_model=MODEL,
        is_se_resnext=IS_SE_RESNEXT,
        pretrained=PRETRAINED,
        blur=BLUR,
        self_attention=SELF_ATTENTION,
        bottle=BOTTLE,
        y_range=Y_RANGE,
        imsize=(SZ, SZ),
        n_classes=NUM_CLASSES,
        loss_func=CRITERION if TRAIN_MODE else None,
        metrics=[METRIC],
        device=DEVICE,
        model_dir=LOGGING_FOLDER
    )

    if HS_MODEL is not None:
        learn.model.load_state_dict(torch.load(HS_MODEL)['model'])

    set_BN_momentum(learn.model, batch_size=BATCH_SIZE)
    learn.clip_grad(1.)

    # callbacks
    csv_logger = CSVLogger(learn=learn, filename=f'{LOGGING_FOLDER}/fit_trace', append=True)
    early_stopping = EarlyStoppingCallback(learn=learn, monitor='dice', patience=PATIENCE)
    save_model = SaveModelCallback(learn=learn, monitor='dice', name='best_model')
    acc_grad = AccumulateStep(learn, 64 // BATCH_SIZE)

    # # find optimal LR
    # learn.lr_find(stop_div=True, num_it=100)
    # learn.recorder.plot(suggestion=True)
    # opt_lr = learn.recorder.min_grad_lr
    # print(f'Initial optimal lr: {opt_lr}')

    if TRAIN_MODE:

        if HS_MODEL is None:

            opt_lr = 0.001

            # fit with frozen
            learn.fit_one_cycle(
                cyc_len=6,
                max_lr=opt_lr,
                callbacks=[
                    acc_grad,
                    csv_logger,
                    early_stopping,
                    save_model
                ]
            )

            # fit entire model with saving on the best epoch
            learn.unfreeze()
            learn.fit_one_cycle(
                cyc_len=12,
                max_lr=slice(opt_lr / 80, opt_lr / 2),
                callbacks=[
                    acc_grad,
                    csv_logger,
                    early_stopping,
                    save_model
                ]
            )

        opt_lr = 0.0001
        learn.loss_func = CRITERION_ADD

        # fit entire model with saving on the best epoch
        learn.unfreeze()
        learn.fit_one_cycle(
            cyc_len=12,
            max_lr=slice(opt_lr / 80, opt_lr / 2),
            callbacks=[
                acc_grad,
                csv_logger,
                early_stopping,
                save_model
            ]
        )

        learn.freeze()

    # =============================================================================

    # # get probs for ChexModel Validation
    # valid_probs = pd.read_csv(VALID_PROBS, sep=';', header=0, index_col='ImageId')
    # ids = [str(o.stem) + '.png' for o in data.valid_ds.items]
    # probs = valid_probs.loc[ids].values

    # get TTA prediction on validation
    preds, acts = pred_with_flip(learn, ds_type=DatasetType.Valid, probs=None)
    # preds, acts = pred_tta(learn, ds_type=DatasetType.Valid)

    # find optimal thresolds on validation
    best_dsc, pixel_thr, prob_thr = fit_optimal_thresholds(preds, acts)
    print(f'Best Dice = {best_dsc} with PROB_THRESHOLD = {prob_thr} and PIXEL_THRESHOLD = {pixel_thr} ')

    # # get probs for ChexModel Test
    # test_probs = pd.read_csv(TEST_PROBS, sep=';', header=0, index_col='ImageId')
    # ids = [str(o.stem) + '.png' for o in data.test_ds.items]
    # probs = test_probs.loc[ids].values

    # get TTA prediction on test
    preds, _ = pred_with_flip(learn, ds_type=DatasetType.Test, probs=None)
    # preds, _ = pred_tta(learn, ds_type=DatasetType.Test)

    # prepare submission
    rles = prepare_submission(preds, prob_threshold=prob_thr, pixel_threshold=pixel_thr)

    # save submission file
    ids = [o.stem for o in data.test_ds.items]
    sub = pd.DataFrame({
        'ImageId': ids,
        'EncodedPixels': rles
    }, columns=['ImageId', 'EncodedPixels'])
    sub.loc[sub.EncodedPixels == '', 'EncodedPixels'] = '-1'
    sub.to_csv(f'{SUBMISSIONS_FOLDER}/{SUBMISSION_FILE}', sep=',', header=True, index=False)
