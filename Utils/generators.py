import cv2
from torchvision import transforms
from fastai.vision import *

from Utils.preprocessing import Preprocessing

IMG_SIZE = 256
NORM = 255.0
PREPROCESSING = Preprocessing(
    enhance_contrast=True,
    median_blur=False,
    blur_kernel=3
)


def grey2rgb(image):

    return np.concatenate([image[np.newaxis, :] for _ in range(3)], axis=0)


class SIIMDataset(data.Dataset):

    def __init__(self, img_ids, xray_path, is_test=True, mask_path=None, img_size=IMG_SIZE,
                 is_prob=False, preprocessing=None, img_augmentation=None,
                 mask_augmentation=None, imagenet_normalization=False):

        self.img_ids = img_ids
        self.xray_path = xray_path
        self.mask_path = mask_path
        self.is_test = is_test
        # self.img_size = img_size
        self.is_prob = is_prob
        self.preprocessing = preprocessing
        self.imagenet_normalization = imagenet_normalization

        if img_augmentation is not None:
            self.img_augmentation = img_augmentation.localize_random_state_()
        else:
            self.img_augmentation = None

        if img_augmentation is not None and mask_augmentation is not None:
            self.mask_augmentation = mask_augmentation.copy_random_state(img_augmentation, matching='name')
        else:
            self.mask_augmentation = None

    def __getitem__(self, idx):

        img_path = os.path.join(self.xray_path, self.img_ids[idx] + '.png')

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if self.preprocessing is not None:
            img = self.preprocessing(img)

        img_aug = None if self.img_augmentation is None else self.img_augmentation.to_deterministic()
        mask_aug = None if self.mask_augmentation is None else self.mask_augmentation.to_deterministic()

        grey_shape = img.shape
        if img_aug is not None:
            img = img_aug(images=[img[:, :, np.newaxis]])[0]
            img = img.reshape(grey_shape)

        img = grey2rgb(img) / NORM

        img = torch.as_tensor(img, dtype=torch.float32)

        if self.imagenet_normalization:
            img = transforms.Normalize([0.540, 0.540, 0.540], [0.264, 0.264, 0.264])(img)

        if self.is_test:
            return img
        else:

            mask_path = os.path.join(self.mask_path, self.img_ids[idx] + '.png')
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_aug is not None:
                mask = mask_aug(images=[mask[:, :, np.newaxis]])[0]
                mask = np.where(mask >= 128, 1, 0)
                mask = np.moveaxis(mask, -1, 0)
            else:
                mask = mask / NORM
                mask = mask[np.newaxis]

            mask = np.squeeze(mask, axis=0)
            mask = torch.as_tensor(mask, dtype=torch.long)

            # mask = torch.as_tensor(mask, dtype=torch.float)

            if self.is_prob:
                return img, (mask.sum() > 0).float()
            else:
                return img, mask

    def __len__(self):

        return len(self.img_ids)


class CheXDataset(data.Dataset):

    def __init__(self, img_ids, xray_path, is_test=True, img_size=IMG_SIZE,
                 preprocessing=None, img_augmentation=None, imagenet_normalization=False):

        self.img_ids = img_ids
        self.xray_path = xray_path
        self.is_test = is_test
        self.img_size = img_size
        self.preprocessing = preprocessing
        self.imagenet_normalization = imagenet_normalization

        if img_augmentation is not None:
            self.img_augmentation = img_augmentation.localize_random_state_()
        else:
            self.img_augmentation = None

    def __getitem__(self, idx):

        img_path = os.path.join(self.xray_path, self.img_ids.loc[idx, 'ImageId'])

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if self.preprocessing is not None:
            img = self.preprocessing(img)

        img_aug = None if self.img_augmentation is None else self.img_augmentation.to_deterministic()

        grey_shape = img.shape
        if img_aug is not None:
            img = img_aug(images=[img[:, :, np.newaxis]])[0]
            img = img.reshape(grey_shape)

        img = grey2rgb(img) / NORM

        img = torch.as_tensor(img, dtype=torch.float32)

        if self.imagenet_normalization:
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        if self.is_test:
            return img
        else:

            target = self.img_ids.loc[idx, 'Target']
            target = torch.as_tensor(target, dtype=torch.float)

            return img, target

    def __len__(self):

        return len(self.img_ids)


if __name__ == "__main__":

    TRAIN_RLE_FILE = '../Data/train_rle.csv'
    TRAIN_XRAY_PATH = '../Data/train_small_xray/'
    TRAIN_MASK_PATH = '../Data/train_small_mask/'

    img_ids = pd.read_csv(TRAIN_RLE_FILE, sep=', ', header=0, index_col='ImageId').index.unique().values

    dataset = SIIMDataset(img_ids[:10], TRAIN_XRAY_PATH, is_test=False, mask_path=TRAIN_MASK_PATH)
    dataloader = data.DataLoader(dataset, batch_size=10, shuffle=True)

    for img, mask in dataloader:
        print(img.shape, mask.shape)

    dataset = SIIMDataset(img_ids[:10], TRAIN_XRAY_PATH, is_test=True)
    dataloader = data.DataLoader(dataset, batch_size=10, shuffle=True)

    for img in dataloader:
        print(img.shape)
