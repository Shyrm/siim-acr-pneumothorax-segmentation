from fastai.vision import *
from fastai.callbacks import CSVLogger, EarlyStoppingCallback, SaveModelCallback

import pretrainedmodels

from Utils.utils import seed_everything
from Utils.metrics import acc
from Utils.utils import AccumulateStep, set_BN_momentum
from Utils.training_fastai import cnn_learner_e
from Utils.generators_fastai import get_data_prob
import torchvision


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

# DATA_PATH = './Data/ProbData'
DATA_PATH = './Data'

TRAIN = f'{DATA_PATH}/Images'
LABELS = f'{DATA_PATH}/Targets.csv'
VALID_IDX = f'{DATA_PATH}/ValidIdx.csv'


PRETRAINED = None
IS_SE_RESNEXT = False
ENCODER = torchvision.models.resnet34
# ENCODER = torchvision.models.densenet121
# ENCODER = pretrainedmodels.se_resnext50_32x4d


NUM_CLASSES = 2
Y_RANGE = None
NUM_EPOCHS = 250
PATIENCE = 8
NUM_WORKERS = 30

MODELS_FOLDER = './FittedModels'
LOGGING_FOLDER = f'{MODELS_FOLDER}/ProbModel_v01'

CRITERION = CrossEntropyFlat(axis=1)
METRIC = acc(threshold=0.5)


if __name__ == '__main__':

    # create folders if not exists
    if not os.path.exists(MODELS_FOLDER):
        os.makedirs(MODELS_FOLDER)

    if not os.path.exists(LOGGING_FOLDER):
        os.makedirs(LOGGING_FOLDER)

    data = get_data_prob(
        train_folder=TRAIN,
        labels_file=LABELS,
        valid_idx=pd.read_csv(VALID_IDX, sep=';', header=0)['Idx'].values,
        img_size=(SZ, SZ),
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        normalization=STATS
    )

    # ============================== model learning ==============================
    learn = cnn_learner_e(
        data=data,
        base_arch=ENCODER,
        pretrained=PRETRAINED,
        is_se_resnext=IS_SE_RESNEXT,
        metrics=[METRIC],
        device=DEVICE,
        model_dir=LOGGING_FOLDER
    )

    set_BN_momentum(learn.model, batch_size=BATCH_SIZE)
    learn.clip_grad(1.)

    # callbacks
    csv_logger = CSVLogger(learn=learn, filename=f'{LOGGING_FOLDER}/fit_trace', append=True)
    early_stopping = EarlyStoppingCallback(learn=learn, monitor='accuracy', patience=PATIENCE)
    save_model = SaveModelCallback(learn=learn, monitor='accuracy', name='best_model')
    acc_grad = AccumulateStep(learn, 64 // BATCH_SIZE)

    opt_lr = 0.001

    # fit with frozen
    learn.fit_one_cycle(
        cyc_len=3,
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
