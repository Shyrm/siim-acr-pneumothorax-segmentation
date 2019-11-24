import os
import cv2
import glob2
import pydicom
import pandas as pd
import numpy as np
from skimage import exposure
from Utils.mask_functions import rle2mask
from tqdm import tqdm
from shutil import copyfile

MIN_MASK_SIZE = 20
VALIDATION_SIZE = 0.2

MASK_DATASET_OUTPUT = './Data/MaskData'
PROB_DATASET_OUTUPT = './Data/ProbData'

MASK_DATA_TEST = './RawData/dicom-images-test'
MASK_DATA_TRAIN = './RawData/dicom-images-train'
MASK_DATA_RLE = './RawData/train-rle.csv'
MASK_DATA_TRAIN_META = './RawData/train_meta.csv'
MASK_DATA_TEST_META = './RawData/test_meta.csv'

PROB_DATA = './RawData/CheXRay'
PROB_DATA_TARGETS = './RawData/Data_Entry_2017.csv'

MASK_DATA_TRAIN_OUTPUT = f'{MASK_DATASET_OUTPUT}/train'
MASK_DATA_TRAIN_MASK_OUTPUT = f'{MASK_DATASET_OUTPUT}/mask'
MASK_DATA_TEST_OUTPUT = f'{MASK_DATASET_OUTPUT}/test'
MASK_DATA_STATISTICS_OUTPUT = f'{MASK_DATASET_OUTPUT}/Statistics.csv'
MASK_DATA_TRAIN_META_OUTPUT = f'{MASK_DATASET_OUTPUT}/train_meta.csv'
MASK_DATA_TEST_META_OUTPUT = f'{MASK_DATASET_OUTPUT}/test_meta.csv'
MASK_DATA_VALID_INDX_OUTPUT = f'{MASK_DATASET_OUTPUT}/ValidIdx.csv'

PROB_DATA_TRAIN_OUTPUT = f'{PROB_DATASET_OUTUPT}/Images'
PROB_DATA_STATISTICS_OUTPUT = f'{PROB_DATASET_OUTUPT}/Statistics.csv'
PROB_DATA_TARGETS_OUTPUT = f'{PROB_DATASET_OUTUPT}/Targets.csv'
PROB_DATA_META_OUTPUT = f'{PROB_DATASET_OUTUPT}/Meta.csv'
PROB_DATA_VALID_INDX_OUTPUT = f'{PROB_DATASET_OUTUPT}/ValidIdx.csv'


mask_sz = 1024
prob_sz = 1024
sz0 = 1024


def convert_image(filename, output_folder, sz, is_dicom=True, sz0=sz0, add_contrast=True):

    if is_dicom:
        ds = pydicom.read_file(str(filename))
        img = ds.pixel_array
    else:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    if sz != sz0:
        img = cv2.resize(img, (sz, sz))

    if add_contrast:
        img = exposure.equalize_adapthist(img)  # contrast correction

    # image statistics
    x_tot = img.mean()
    x2_tot = (img**2).mean()

    if add_contrast:
        img = (img * 255).clip(0, 255).astype(np.uint8)

    name = filename.split('/')[-1][:-4] + '.png'

    cv2.imwrite(f'{output_folder}/{name}', img)

    return x_tot, x2_tot


def convert_mask(rle, index, output_folder, sz, sz0=sz0):

    img = np.zeros((sz0, sz0))
    if type(rle) != str or (type(rle) == str and rle != '-1'):

        if type(rle) == str:
            rle = [rle]
        else:
            rle = rle.tolist()

        for mask in rle:
            img += rle2mask(mask, sz0, sz0).T

    if sz != sz0:
        img = cv2.resize(img, (sz, sz))

    name = index + '.png'

    cv2.imwrite(f'{output_folder}/{name}', img)


if __name__ == '__main__':

    # ============================ create all necessary folders =====================================

    if not os.path.exists(MASK_DATA_TRAIN_OUTPUT):
        os.makedirs(MASK_DATA_TRAIN_OUTPUT)

    if not os.path.exists(MASK_DATA_TRAIN_MASK_OUTPUT):
        os.makedirs(MASK_DATA_TRAIN_MASK_OUTPUT)

    if not os.path.exists(MASK_DATA_TEST_OUTPUT):
        os.makedirs(MASK_DATA_TEST_OUTPUT)

    if not os.path.exists(PROB_DATA_TRAIN_OUTPUT):
        os.makedirs(PROB_DATA_TRAIN_OUTPUT)

    # ============================ create dataset for mask learning =================================

    print('Preparing train for mask data...')
    train_dcm_list = glob2.glob(os.path.join(MASK_DATA_TRAIN, '**/*.dcm'))
    rle_data = pd.read_csv(MASK_DATA_RLE, sep=', ', header=0, index_col='ImageId')
    idxs = set(rle_data.index)

    statistics = []
    for file in tqdm(train_dcm_list):

        idx = file.split('/')[-1][:-4]  # get index of the image

        # ignore images without masks
        if idx not in idxs:
            continue

        img_mean, img2_mean = convert_image(
            filename=file,
            sz=mask_sz,
            output_folder=MASK_DATA_TRAIN_OUTPUT
        )

        convert_mask(
            rle=rle_data.loc[idx, 'EncodedPixels'],
            index=idx,
            sz=mask_sz,
            output_folder=MASK_DATA_TRAIN_MASK_OUTPUT
        )

        statistics.append([img_mean, img2_mean])

    print('Preparing test for mask data...')
    test_dcm_list = glob2.glob(os.path.join(MASK_DATA_TEST, '**/*.dcm'))

    for file in tqdm(test_dcm_list):

        idx = file.split('/')[-1][:-4]  # get index of the image

        img_mean, img2_mean = convert_image(
            filename=file,
            sz=mask_sz,
            output_folder=MASK_DATA_TEST_OUTPUT
        )

        statistics.append([img_mean, img2_mean])

    # store statistics for mask data separately
    pd.DataFrame(statistics, columns=['mu', '2mu']).to_csv(MASK_DATA_STATISTICS_OUTPUT, sep=';', header=True, index=False)

    print('Preparing CheXRay data...')

    targets_raw = pd.read_csv(PROB_DATA_TARGETS, sep=',', header=0)
    targets_raw.index = targets_raw['Image Index']

    print('Processing original CheXRay...')
    targets = []
    for f in os.listdir(PROB_DATA):

        # skip archives
        if '.tar.gz' in f:
            continue

        print(f)
        direct_folder = f'{PROB_DATA}/{f}/images'
        for file in tqdm(os.listdir(direct_folder)):

            img_mean, img2_mean = convert_image(
                filename=f'{direct_folder}/{file}',
                output_folder=PROB_DATA_TRAIN_OUTPUT,
                is_dicom=False,
                sz=prob_sz
            )

            statistics.append([img_mean, img2_mean])

            targets.append([
                file,
                1 if 'Pneumothorax' in targets_raw.loc[file, 'Finding Labels'] else 0
            ])

    # store full statistics for prob data
    pd.DataFrame(statistics, columns=['mu', '2mu']).to_csv(PROB_DATA_STATISTICS_OUTPUT,
                                                           sep=';', header=True, index=False)

    print(f'Adding data from mask dataset...')
    for file in tqdm(os.listdir(MASK_DATA_TRAIN_OUTPUT)):

        mask = cv2.imread(f'{MASK_DATA_TRAIN_MASK_OUTPUT}/{file}', cv2.IMREAD_GRAYSCALE)

        targets.append([
            file,
            1 if np.sum(mask) >= MIN_MASK_SIZE else 0
        ])

        if mask_sz == prob_sz:

            dest = f'{PROB_DATA_TRAIN_OUTPUT}/{file}'
            source = f'{MASK_DATA_TRAIN_OUTPUT}/{file}'
            copyfile(source, dest)

        else:

            _, _ = convert_image(
                filename=f'{MASK_DATA_TRAIN_OUTPUT}/{file}',
                output_folder=PROB_DATA_TRAIN_OUTPUT,
                sz=prob_sz,
                is_dicom=False,
                sz0=mask_sz,
                add_contrast=False
            )

    # store targets for prob data
    pd.DataFrame(targets, columns=['Image', 'IsPneumo']).to_csv(PROB_DATA_TARGETS_OUTPUT,
                                                                sep=';', header=True, index=False)

    # store metadata for mask test
    copyfile(MASK_DATA_TEST_META, MASK_DATA_TEST_META_OUTPUT)

    # store metadata from mask train
    mask_meta_train = pd.read_csv(MASK_DATA_TRAIN_META, sep=';', header=0)
    valid_idxs = [e[:-4] for e in os.listdir(MASK_DATA_TRAIN_OUTPUT)]
    mask_meta_train = mask_meta_train[mask_meta_train['ImageId'].isin(valid_idxs)].copy()

    mask_meta_train.to_csv(MASK_DATA_TRAIN_META_OUTPUT, sep=';', header=True, index=False)

    # store metadata for CheXRay
    targets_raw = targets_raw[['Image Index', 'Patient Age', 'Patient Gender', 'View Position', 'OriginalImagePixelSpacing[x', 'y]']]
    targets_raw['PixelSpacing'] = targets_raw.apply(lambda row: f"['{row['OriginalImagePixelSpacing[x']}', '{row['y]']}']", axis=1)
    targets_raw.drop(['OriginalImagePixelSpacing[x', 'y]'], axis=1, inplace=True)
    targets_raw.rename(columns={'Patient Age': 'PatientAge',
                                'Patient Gender': 'PatientSex',
                                'View Position': 'ViewPosition',
                                'Image Index': 'ImageId'}, inplace=True)

    mask_meta_train = mask_meta_train[list(targets_raw.columns)]
    targets_raw = pd.concat([targets_raw, mask_meta_train])
    targets_raw.to_csv(PROB_DATA_META_OUTPUT, sep=';', header=True, index=False)

    # ================================ create and store validation index =========================================
    DATA_PATH = './Data'

    train_data = pd.read_csv(MASK_DATA_TRAIN_META_OUTPUT, sep=';', header=0)
    ids = train_data['ImageId'].values

    np.random.seed(42)

    train_idx = np.random.choice(ids, size=int((1. - VALIDATION_SIZE) * len(ids)), replace=False)
    validation_idx = [i for i in ids if i not in train_idx]

    pd.DataFrame(validation_idx, columns=['Idx']).to_csv(MASK_DATA_VALID_INDX_OUTPUT, sep=';', header=True, index=False)
    pd.DataFrame(validation_idx, columns=['Idx']).to_csv(PROB_DATA_VALID_INDX_OUTPUT, sep=';', header=True, index=False)

    # ================================ add binary labels to mask data ===========================================
    labels = pd.read_csv('./Data/ProbData/Targets.csv', sep=';', header=0)

    labels['len'] = labels['Image'].apply(len)
    labels = labels[labels['len'] > 20].copy().reset_index(drop=True)

    labels[['Image', 'IsPneumo']].to_csv('./Data/MaskData/Targets.csv', sep=';', header=True, index=False)
