import glob
import os

import numpy as np
import pytorch_lightning as pl
from monai.data import CacheDataset, DataLoader, Dataset, PersistentDataset, partition_dataset
from monai.transforms import (
    AddChanneld,
    Compose,
    CropForegroundd,
    DeleteItemsd,
    FgBgToIndicesd,
    Lambdad,
    LoadImage,
    LoadImaged,
    MapLabelValued,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    SpatialPadd,
    ToTensord,
    Transpose,
)


class LUNA16DataModule(pl.LightningDataModule):

    class_weight = np.asarray([0.13143768, 0.86856232])

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
        **kwargs,
    ):
        super().__init__()
        self.root_dir = root_dir
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
                    LoadImaged(keys=["image", "label"], reader="ITKReader"),
                    AddChanneld(keys=["image", "label"]),
                    Lambdad(keys=["image", "label"], func=Transpose((0, 3, 2, 1))),
                    Orientationd(keys=["image", "label"], axcodes="RAI"),
                    MapLabelValued(
                        keys=["label"], orig_labels=[0, 3, 4, 5], target_labels=[0, 1, 1, 0], dtype=np.float32
                    ),
                    ScaleIntensityRanged(keys=["image"], a_min=-1024, a_max=3072, b_min=0.0, b_max=1.0, clip=True),
                    CropForegroundd(keys=["image", "label"], source_key="image", margin=0),
                    SpatialPadd(keys=["image", "label"], spatial_size=train_patch_size, mode="edge"),
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
                    LoadImaged(keys=["image", "label"], reader="ITKReader"),
                    AddChanneld(keys=["image", "label"]),
                    Lambdad(keys=["image", "label"], func=Transpose((0, 3, 2, 1))),
                    Orientationd(keys=["image", "label"], axcodes="RAI"),
                    MapLabelValued(
                        keys=["label"], orig_labels=[0, 3, 4, 5], target_labels=[0, 1, 1, 0], dtype=np.float32
                    ),
                    ScaleIntensityRanged(keys=["image"], a_min=-1024, a_max=3072, b_min=0.0, b_max=1.0, clip=True),
                    CropForegroundd(keys=["image", "label"], source_key="image", margin=0),
                    ToTensord(keys=["image", "label"]),
                ]
            )
        else:
            self.val_transforms = val_transforms

    def _load_data_dicts(self, train=True):
        images = sorted(glob.glob(os.path.join(self.root_dir, "imgs", "*.mhd")))

        if train:
            labels = sorted(glob.glob(os.path.join(self.root_dir, "segs", "*.mhd")))
            data_dicts = [{"image": img_name, "label": label_name} for img_name, label_name in zip(images, labels)]
            data_dicts_list = partition_dataset(data_dicts, num_partitions=4, shuffle=True, seed=0)
            train_dicts, val_dicts = [], []
            for i, data_dict in enumerate(data_dicts_list):
                if i == self.fold:
                    val_dicts.extend(data_dict)
                else:
                    train_dicts.extend(data_dict)
            return train_dicts, val_dicts
        else:
            data_dicts = [{"image": img_name} for img_name in images]
            return data_dicts

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
                    data=val_data_dicts,
                    transform=self.val_transforms,
                    cache_rate=self.cache_rate,
                    num_workers=self.num_workers,
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
                data=val_data_dicts,
                transform=self.val_transforms,
                cache_rate=self.cache_rate,
                num_workers=self.num_workers,
            )

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, pin_memory=False, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=1, num_workers=self.num_workers)

    def test_dataloader(self):
        pass

    def calculate_class_weight(self):
        class_weight = []
        for label_name in sorted(glob.glob(os.path.join(self.root_dir, "segs", "*.mhd"))):
            label = LoadImage(reader="ITKReader", image_only=True)(label_name)
            # Trachea
            label[label == 5] = 0
            label[label > 1] = 1

            _, counts = np.unique(label, return_counts=True)
            counts = np.sum(counts) / counts
            # Normalize
            counts = counts / np.sum(counts)
            class_weight.append(counts)

        class_weight = np.asarray(class_weight)
        class_weight = np.mean(class_weight, axis=0)
        print("Class weight: ", class_weight)

    def calculate_class_percentage(self):
        class_percentage = []
        for label_name in sorted(glob.glob(os.path.join(self.root_dir, "segs", "*.mhd"))):
            label = LoadImage(reader="ITKReader", image_only=True)(label_name)
            # Trachea
            label[label == 5] = 0
            label[label > 1] = 1

            _, counts = np.unique(label, return_counts=True)
            # Normalize
            counts = counts / np.sum(counts)
            class_percentage.append(counts)

        class_percentage = np.asarray(class_percentage)
        class_percentage = np.mean(class_percentage, axis=0)
        print("Class Percentage: ", class_percentage)


if __name__ == "__main__":
    data_module = LUNA16DataModule(
        root_dir="/home/ubuntu/",
        train_patch_size=[96, 96, 96],
        num_samples=2,
        batch_size=1,
        cache_dir="/home/ubuntu/cache_dir",
    )
    # data_module.calculate_class_weight()
    # data_module.calculate_class_percentage()
