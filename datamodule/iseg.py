import glob
import os

import numpy as np
import pytorch_lightning as pl
from monai.data import CacheDataset, DataLoader, Dataset, PersistentDataset
from monai.transforms import (
    AddChanneld,
    Compose,
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


class ISeg2017DataModule(pl.LightningDataModule):

    class_weight = np.asarray([0.01254255, 0.44403431, 0.21026247, 0.33316067])

    def __init__(
        self,
        root_dir=".",
        train_patch_size=(32, 32, 32),
        num_samples=32,
        batch_size=1,
        cache_rate=None,
        cache_dir=None,
        num_workers=4,
        balance_sampling=True,
        n_val_replication=0,
        train_transforms=None,
        val_transforms=None,
        **kwargs,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.cache_rate = cache_rate
        self.num_workers = num_workers
        self.n_val_replication = n_val_replication

        if balance_sampling:
            pos = neg = 0.5
        else:
            pos = np.sum(self.class_weight[1:])
            neg = self.class_weight[0]

        if train_transforms is None:
            self.train_transforms = Compose(
                [
                    LoadImaged(keys=["image", "label"], reader="ITKReader"),
                    AddChanneld(keys=["label"]),
                    Lambdad(keys=["image", "label"], func=Transpose((0, 3, 2, 1))),
                    Orientationd(keys=["image", "label"], axcodes="RAI"),
                    MapLabelValued(
                        keys=["label"], orig_labels=[0, 10, 150, 250], target_labels=[0, 1, 2, 3], dtype=np.float32
                    ),
                    SpatialPadd(keys=["image", "label"], spatial_size=train_patch_size, mode="edge"),
                    ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=1000.0, b_min=0.0, b_max=1.0, clip=True),
                    FgBgToIndicesd(keys=["label"]),
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
                    AddChanneld(keys=["label"]),
                    Lambdad(keys=["image", "label"], func=Transpose((0, 3, 2, 1))),
                    Orientationd(keys=["image", "label"], axcodes="RAI"),
                    MapLabelValued(keys=["label"], orig_labels=[0, 10, 150, 250], target_labels=[0, 1, 2, 3]),
                    ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=1000.0, b_min=0.0, b_max=1.0, clip=True),
                    ToTensord(keys=["image", "label"]),
                ]
            )
        else:
            self.val_transforms = val_transforms

    def _load_data_dicts(self, data_dir, n_replication=0, train=True):
        t1_images = sorted(glob.glob(os.path.join(data_dir, "*T1.hdr")))
        t2_images = sorted(glob.glob(os.path.join(data_dir, "*T2.hdr")))
        if train:
            labels = sorted(glob.glob(os.path.join(data_dir, "*label.hdr")))
            data_dicts = [
                {"image": [t1_name, t2_name], "label": label_name}
                for t1_name, t2_name, label_name in zip(t1_images, t2_images, labels)
            ]
        else:
            data_dicts = [{"image": [t1_name, t2_name]} for t1_name, t2_name in zip(t1_images, t2_images)]

        ori_data_dicts = data_dicts.copy()
        for _ in range(n_replication):
            data_dicts.extend(ori_data_dicts)
        return data_dicts

    def setup(self, stage=None):
        if stage in (None, "fit"):
            train_data_dicts = self._load_data_dicts(os.path.join(self.root_dir, "domainA"))
            val_data_dicts = self._load_data_dicts(
                os.path.join(self.root_dir, "domainA_val"), n_replication=self.n_val_replication
            )

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
            val_data_dicts = self._load_data_dicts(os.path.join(self.root_dir, "domainA_val"))
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
        class_weight = []
        for label_name in sorted(glob.glob(os.path.join(self.root_dir, "domainA", "*label.hdr"))):
            label = LoadImage(reader="ITKReader", image_only=True)(label_name)

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
        for label_name in sorted(glob.glob(os.path.join(self.root_dir, "domainA", "*label.hdr"))):
            label = LoadImage(reader="ITKReader", image_only=True)(label_name)

            _, counts = np.unique(label, return_counts=True)
            # Normalize
            counts = counts / np.sum(counts)
            class_percentage.append(counts)

        class_percentage = np.asarray(class_percentage)
        class_percentage = np.mean(class_percentage, axis=0)
        print("Class Percentage: ", class_percentage)


if __name__ == "__main__":
    data_module = ISeg2017DataModule(root_dir="/home/ubuntu/")
    # data_module.calculate_class_weight()
    data_module.calculate_class_percentage()
