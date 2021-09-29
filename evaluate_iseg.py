import argparse

import numpy as np
import torch
from datamodule.iseg import ISeg2017DataModule
from module.segcaps import SegCaps2D, SegCaps3D
from module.ucaps import UCaps3D
from module.unet import UNetModule
from monai.data import NiftiSaver, decollate_batch
from monai.metrics import ConfusionMatrixMetric, DiceMetric
from monai.transforms import (  # GibbsNoised,
    AddChanneld,
    AsDiscrete,
    Compose,
    EnsureType,
    Lambdad,
    LoadImaged,
    MapLabelValue,
    MapLabelValued,
    Orientationd,
    Rotated,
    ScaleIntensityRanged,
    ToNumpyd,
    ToTensord,
    Transpose,
)
from monai.utils import set_determinism
from pytorch_lightning import Trainer

# from torchio.transforms import RandomBiasField, RandomGhosting, RandomMotion, RandomSpike
from tqdm import tqdm


def print_metric(metric_name, scores):
    mean_score = np.mean(scores)
    print("-------------------------------")
    print("Validation {} score mean: {:4f}".format(metric_name, mean_score))
    for i, score in enumerate(scores):
        print("Validation {} score class {}: {:4f}".format(metric_name, i + 1, score))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="/root")
    parser.add_argument("--save_image", type=int, default=0, help="Save image or not")

    parser = Trainer.add_argparse_args(parser)

    # Validation config
    val_parser = parser.add_argument_group("Validation config")
    val_parser.add_argument("--output_dir", type=str, default="/root")
    val_parser.add_argument("--model_name", type=str, default="ucaps", help="ucaps / segcaps-2d / segcaps-3d / unet")
    val_parser.add_argument(
        "--checkpoint_path", type=str, default="", help='/path/to/trained_model. Set to "" for none.'
    )
    val_parser.add_argument("--rotate_angle", type=int, default=0)
    val_parser.add_argument("--axis", type=str, default="z", help="z/y/x or all")

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    # let the model add what it wants
    if temp_args.model_name == "ucaps":
        parser, model_parser = UCaps3D.add_model_specific_args(parser)
    elif temp_args.model_name == "segcaps-2d":
        parser, model_parser = SegCaps2D.add_model_specific_args(parser)
    elif temp_args.model_name == "segcaps-3d":
        parser, model_parser = SegCaps3D.add_model_specific_args(parser)
    elif temp_args.model_name == "unet":
        parser, model_parser = UNetModule.add_model_specific_args(parser)

    args = parser.parse_args()
    dict_args = vars(args)

    print("Validation config:")
    for a in val_parser._group_actions:
        print("\t{}:\t{}".format(a.dest, dict_args[a.dest]))

    # Improve reproducibility
    set_determinism(seed=0)

    rotate_angle = args.rotate_angle * np.pi / 180
    if args.axis == "x":
        rotate_transform = Rotated(
            keys=["image", "label"], angle=(rotate_angle, 0, 0), keep_size=False, mode=("bilinear", "nearest")
        )
    elif args.axis == "y":
        rotate_transform = Rotated(
            keys=["image", "label"], angle=(0, rotate_angle, 0), keep_size=False, mode=("bilinear", "nearest")
        )
    elif args.axis == "z":
        rotate_transform = Rotated(
            keys=["image", "label"], angle=(0, 0, rotate_angle), keep_size=False, mode=("bilinear", "nearest")
        )
    elif args.axis == "all":
        rotate_transform = Rotated(
            keys=["image", "label"],
            angle=(rotate_angle, rotate_angle, rotate_angle),
            keep_size=False,
            mode=("bilinear", "nearest"),
        )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], reader="ITKReader"),
            AddChanneld(keys=["label"]),
            Lambdad(keys=["image", "label"], func=Transpose((0, 3, 2, 1))),
            Orientationd(keys=["image", "label"], axcodes="RAI"),
            rotate_transform,
            # GibbsNoised(keys=['image'], alpha=0.5, as_tensor_output=True),
            ToTensord(keys=["image"]),
            # RandomMotion(keys=["image"], image_interpolation='bspline'),
            # RandomGhosting(keys=["image"]),
            # RandomSpike(keys=['image']),
            # RandomBiasField(keys=['image']),
            ToNumpyd(keys=["image"]),
            ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=1000.0, b_min=0.0, b_max=1.0, clip=True),
            MapLabelValued(
                keys=["label"], orig_labels=[0, 10, 150, 250], target_labels=[0, 1, 2, 3], dtype=np.float32
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )

    # Prepare dataset
    data_module = ISeg2017DataModule(args.root_dir, cache_rate=1.0, val_transforms=val_transforms)
    data_module.setup("validate")
    val_loader = data_module.val_dataloader()
    val_batch_size = 1

    # Load trained model

    if args.checkpoint_path != "":
        if args.model_name == "ucaps":
            net = UCaps3D.load_from_checkpoint(
                args.checkpoint_path,
                val_patch_size=args.val_patch_size,
                sw_batch_size=args.sw_batch_size,
                overlap=args.overlap,
            )
        elif args.model_name == "unet":
            net = UNetModule.load_from_checkpoint(
                args.checkpoint_path,
                val_patch_size=args.val_patch_size,
                sw_batch_size=args.sw_batch_size,
                overlap=args.overlap,
            )
        elif args.model_name == "segcaps-2d":
            net = SegCaps2D.load_from_checkpoint(
                args.checkpoint_path,
                val_patch_size=args.val_patch_size,
                sw_batch_size=args.sw_batch_size,
                overlap=args.overlap,
            )
        elif args.model_name == "segcaps-3d":
            net = SegCaps3D.load_from_checkpoint(
                args.checkpoint_path,
                val_patch_size=args.val_patch_size,
                sw_batch_size=args.sw_batch_size,
                overlap=args.overlap,
            )
        print("Load trained model!!!")

    # Prediction

    trainer = Trainer.from_argparse_args(args)
    outputs = trainer.predict(net, dataloaders=val_loader)

    # Calculate metric and visualize

    n_classes = net.out_channels
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=True, n_classes=n_classes)])
    save_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=False, n_classes=n_classes)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=True, n_classes=n_classes)])

    map_label = MapLabelValue(target_labels=[0, 10, 150, 250], orig_labels=[0, 1, 2, 3], dtype=np.uint8)

    image_saver = NiftiSaver(
        output_dir=args.output_dir,
        output_postfix=f"{args.rotate_angle}_degree_{args.axis}_axis",
        resample=False,
        data_root_dir=args.root_dir,
    )
    label_saver = NiftiSaver(
        output_dir=args.output_dir,
        output_postfix=f"{args.rotate_angle}_degree_{args.axis}_axis",
        resample=False,
        data_root_dir=args.root_dir,
        output_dtype=np.uint8,
    )
    pred_saver = NiftiSaver(
        output_dir=args.output_dir,
        output_postfix=f"{args.rotate_angle}_degree_{args.axis}_axis_{args.model_name}_prediction",
        resample=False,
        data_root_dir=args.root_dir,
        output_dtype=np.uint8,
    )

    dice_metric = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False)
    precision_metric = ConfusionMatrixMetric(
        include_background=False,
        metric_name="precision",
        compute_sample=True,
        reduction="mean_batch",
        get_not_nans=False,
    )
    sensitivity_metric = ConfusionMatrixMetric(
        include_background=False,
        metric_name="sensitivity",
        compute_sample=True,
        reduction="mean_batch",
        get_not_nans=False,
    )

    for i, data in enumerate(tqdm(val_loader)):
        if args.save_image:
            label_saver.save_batch(
                map_label(data["label"]),
                meta_data={
                    "filename_or_obj": data["label_meta_dict"]["filename_or_obj"],
                    "original_affine": data["label_meta_dict"]["original_affine"],
                    "affine": data["label_meta_dict"]["affine"],
                },
            )
            image_saver.save_batch(
                data["image"][:, [0], ...],
                meta_data={
                    "filename_or_obj": data["image_meta_dict"]["filename_or_obj"],
                    "original_affine": data["image_meta_dict"]["original_affine"],
                    "affine": data["image_meta_dict"]["affine"],
                },
            )

        labels = data["label"]

        val_outputs = outputs[i].cpu()

        if args.save_image:
            pred_saver.save_batch(
                map_label(torch.stack([save_pred(i) for i in decollate_batch(val_outputs)]).cpu()),
                meta_data={
                    "filename_or_obj": data["label_meta_dict"]["filename_or_obj"],
                    "original_affine": data["label_meta_dict"]["original_affine"],
                    "affine": data["label_meta_dict"]["affine"],
                },
            )

        val_outputs = [post_pred(val_output) for val_output in decollate_batch(val_outputs)]
        labels = [post_label(label) for label in decollate_batch(labels)]

        dice_metric(y_pred=val_outputs, y=labels)
        precision_metric(y_pred=val_outputs, y=labels)
        sensitivity_metric(y_pred=val_outputs, y=labels)

    print_metric("dice", dice_metric.aggregate().cpu().numpy())

    print_metric("precision", precision_metric.aggregate()[0].cpu().numpy())

    print_metric("sensitivity", sensitivity_metric.aggregate()[0].cpu().numpy())

    print("Finished Evaluation")
