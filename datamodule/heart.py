import os

import numpy as np
import pytorch_lightning as pl
from monai.data import CacheDataset, DataLoader, Dataset, PersistentDataset, load_decathlon_datalist, partition_dataset
from monai.transforms import (
    AddChanneld,
    Compose,
    CropForegroundd,
    DeleteItemsd,
    FgBgToIndicesd,
    LoadImage,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityd,
    SpatialPadd,
    ToTensord,
)


class HeartDecathlonDataModule(pl.LightningDataModule):

    class_weight = np.asarray([0.00404219, 0.99595781])

    def __init__(
        self,
        root_dir=".",
        fold=0,
        train_patch_size=(32, 32, 32),
        num_samples=32,
        batch_size=1,
        cache_rate=None,
        cache_dir=None,
        num_workers=4,
        balance_sampling=True,
        train_transforms=None,
        val_transforms=None,
        **kwargs
    ):
        super().__init__()
        self.base_dir = root_dir
        self.fold = fold
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.cache_rate = cache_rate
        self.num_workers = num_workers

        if balance_sampling:
            pos = neg = 0.5
        else:
            pos = np.sum(self.class_weight[1:])
            neg = self.class_weight[0]

        if train_transforms is None:
            self.train_transforms = Compose(
                [
                    LoadImaged(keys=["image", "label"], reader="NibabelReader"),
                    AddChanneld(keys=["image", "label"]),
                    Orientationd(keys=["image", "label"], axcodes="LPI"),
                    CropForegroundd(keys=["image", "label"], source_key="image", margin=0),
                    SpatialPadd(keys=["image", "label"], spatial_size=train_patch_size, mode="edge"),
                    ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
                    FgBgToIndicesd(keys=["label"], image_key="image"),
                    RandCropByPosNegLabeld(
                        keys=["image", "label"],
                        label_key="label",
                        spatial_size=train_patch_size,
                        pos=pos,
                        neg=neg,
                        num_samples=num_samples,
                        fg_indices_key="label_fg_indices",
                        bg_indices_key="label_bg_indices",
                    ),
                    DeleteItemsd(keys=["label_fg_indices", "label_bg_indices"]),
                    ToTensord(keys=["image", "label"]),
                ]
            )
        else:
            self.train_transforms = train_transforms

        if val_transforms is None:
            self.val_transforms = Compose(
                [
                    LoadImaged(keys=["image", "label"], reader="NibabelReader"),
                    AddChanneld(keys=["image", "label"]),
                    Orientationd(keys=["image", "label"], axcodes="LPI"),
                    CropForegroundd(keys=["image", "label"], source_key="image", margin=0),
                    ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
                    ToTensord(keys=["image", "label"]),
                ]
            )
        else:
            self.val_transforms = val_transforms

    def _load_data_dicts(self, train=True):
        if train:
            data_dicts = load_decathlon_datalist(
                os.path.join(self.base_dir, "dataset.json"), data_list_key="training", base_dir=self.base_dir
            )
            data_dicts_list = partition_dataset(data_dicts, num_partitions=4, shuffle=True, seed=0)
            train_dicts, val_dicts = [], []
            for i, data_dict in enumerate(data_dicts_list):
                if i == self.fold:
                    val_dicts.extend(data_dict)
                else:
                    train_dicts.extend(data_dict)
            return train_dicts, val_dicts
        else:
            pass

    def setup(self, stage=None):
        if stage in (None, "fit"):
            train_data_dicts, val_data_dicts = self._load_data_dicts()

            if self.cache_rate is not None:
                self.trainset = CacheDataset(
                    data=train_data_dicts,
                    transform=self.train_transforms,
                    cache_rate=self.cache_rate,
                    num_workers=self.num_workers,
                )
                self.valset = CacheDataset(
                    data=val_data_dicts, transform=self.val_transforms, cache_rate=1.0, num_workers=4
                )
            elif self.cache_dir is not None:
                self.trainset = PersistentDataset(
                    data=train_data_dicts, transform=self.train_transforms, cache_dir=self.cache_dir
                )
                self.valset = PersistentDataset(
                    data=val_data_dicts, transform=self.val_transforms, cache_dir=self.cache_dir
                )
            else:
                self.trainset = Dataset(data=train_data_dicts, transform=self.train_transforms)
                self.valset = Dataset(data=val_data_dicts, transform=self.val_transforms)
        elif stage == "validate":
            _, val_data_dicts = self._load_data_dicts()
            self.valset = CacheDataset(
                data=val_data_dicts, transform=self.val_transforms, cache_rate=1.0, num_workers=4
            )

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=1, num_workers=4)

    def test_dataloader(self):
        pass

    def calculate_class_weight(self):
        data_dicts = load_decathlon_datalist(
            os.path.join(self.base_dir, "dataset.json"), data_list_key="training", base_dir=self.base_dir
        )

        class_weight = []
        for data_dict in data_dicts:
            label = LoadImage(reader="NibabelReader", image_only=True)(data_dict["label"])

            _, counts = np.unique(label, return_counts=True)
            counts = np.sum(counts) / counts
            # Normalize
            counts = counts / np.sum(counts)
            class_weight.append(counts)

        class_weight = np.asarray(class_weight)
        class_weight = np.mean(class_weight, axis=0)
        print("Class weight: ", class_weight)

    def calculate_class_percentage(self):
        data_dicts = load_decathlon_datalist(
            os.path.join(self.base_dir, "dataset.json"), data_list_key="training", base_dir=self.base_dir
        )

        class_percentage = []
        for data_dict in data_dicts:
            label = LoadImage(reader="NibabelReader", image_only=True)(data_dict["label"])

            _, counts = np.unique(label, return_counts=True)
            # Normalize
            counts = counts / np.sum(counts)
            class_percentage.append(counts)

        class_percentage = np.asarray(class_percentage)
        class_percentage = np.mean(class_percentage, axis=0)
        print("Class Percentage: ", class_percentage)


if __name__ == "__main__":
    data_module = HeartDecathlonDataModule(root_dir="/home/ubuntu/Task02_Heart")
    # data_module.calculate_class_weight()
    data_module.calculate_class_percentage()
