from fastai.vision import *


class SegmentationLabelList(SegmentationLabelList):
    def open(self, fn):
        return open_mask(fn, div=True)


class SIIMItemList(ImageList):

    _label_cls = SegmentationLabelList


class SIIMLabelLists(LabelLists):

    def transform(self, tfms: Optional[Tuple[TfmList, TfmList]] = (None, None), **kwargs):
        if not tfms: tfms = (None, None)
        assert is_listy(tfms) and len(tfms) == 2
        self.train.transform(tfms[0], **kwargs)
        self.valid.transform(tfms[1], **kwargs)
        kwargs['tfm_y'] = False  # Test data has no labels
        if self.test: self.test.transform(tfms[1], **kwargs)
        return self


def get_data_fastai(train_path, valid_idx, test_path,
                    img_size, batch_size, num_workers,
                    normalization=([0.540, 0.540, 0.540], [0.264, 0.264, 0.264])):

    data = (SIIMItemList.from_folder(train_path)
            .split_by_files([str(idx) + '.png' for idx in valid_idx])
            # .label_from_func(lambda x: str(x).replace('xray', 'mask'), classes=[0, 1])
            .label_from_func(lambda x: str(x).replace('train', 'mask'), classes=[0, 1])
            .add_test(Path(test_path).ls(), label=None))

    data.__class__ = SIIMLabelLists

    data = (data.transform(get_transforms(), size=img_size, tfm_y=True)
            .databunch(path=Path('.'), bs=batch_size, num_workers=num_workers)
            .normalize(normalization)
            )

    return data


def get_data_prob(
        train_folder,
        labels_file,
        valid_idx,
        img_size=(256, 256),
        batch_size=16,
        num_workers=3,
        normalization=([0.540, 0.540, 0.540], [0.264, 0.264, 0.264])
):

    labels = pd.read_csv(labels_file, sep=';', header=0)

    data = (ImageList.from_df(df=labels, path=train_folder)
            .split_by_files([str(idx) + '.png' for idx in valid_idx])
            .label_from_df()
            .transform(get_transforms(), size=img_size, tfm_y=False)
            .databunch(path=Path('.'), bs=batch_size, num_workers=num_workers)
            .normalize(normalization))

    return data
