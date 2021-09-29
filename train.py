from argparse import ArgumentParser

import torch
from datamodule.heart import HeartDecathlonDataModule
from datamodule.hippocampus import HippocampusDecathlonDataModule
from datamodule.iseg import ISeg2017DataModule
from datamodule.luna import LUNA16DataModule
from module.segcaps import SegCaps2D, SegCaps3D
from module.ucaps import UCaps3D
from module.unet import UNetModule
from monai.utils import set_determinism
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


# Call example
# python train3D.py --gpus 1 --model_name UCaps --num_workers 4 --max_epochs 20000 --check_val_every_n_epoch 100 --log_dir=../logs --root_dir=/home/ubuntu/

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="/root")
    parser.add_argument("--cache_rate", type=float, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)

    # Training options
    train_parser = parser.add_argument_group("Training config")
    train_parser.add_argument("--log_dir", type=str, default="/mnt/vinai/logs")
    train_parser.add_argument("--model_name", type=str, default="ucaps", help="ucaps / segcaps-2d / segcaps-3d / unet")
    train_parser.add_argument(
        "--dataset", type=str, default="iseg2017", help="iseg2017 / task02_heart / task04_hippocampus / luna16"
    )
    train_parser.add_argument("--train_patch_size", nargs="+", type=int, default=[32, 32, 32])
    train_parser.add_argument("--fold", type=int, default=0)
    train_parser.add_argument("--num_workers", type=int, default=4)
    train_parser.add_argument("--batch_size", type=int, default=1)
    train_parser.add_argument(
        "--num_samples", type=int, default=1, help="Effective batch size: batch_size x num_samples"
    )
    train_parser.add_argument("--balance_sampling", type=int, default=1)
    train_parser.add_argument("--use_class_weight", type=int, default=0)

    parser = Trainer.add_argparse_args(parser)

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

    print(f"{args.model_name} config:")
    for a in model_parser._group_actions:
        print("\t{}:\t{}".format(a.dest, dict_args[a.dest]))

    print("Training config:")
    for a in train_parser._group_actions:
        print("\t{}:\t{}".format(a.dest, dict_args[a.dest]))

    # Improve reproducibility
    set_determinism(seed=0)
    seed_everything(0, workers=True)

    # Set up datamodule
    if args.dataset == "iseg2017":
        data_module = ISeg2017DataModule(
            **dict_args,
            n_val_replication=args.gpus - 1,
        )
    elif args.dataset == "task02_heart":
        data_module = HeartDecathlonDataModule(
            **dict_args,
        )
    elif args.dataset == "task04_hippocampus":
        data_module = HippocampusDecathlonDataModule(
            **dict_args,
        )
    elif args.dataset == "luna16":
        data_module = LUNA16DataModule(
            **dict_args,
        )
    else:
        pass

    # initialise the LightningModule

    if args.use_class_weight:
        class_weight = torch.tensor(data_module.class_weight).float()
    else:
        class_weight = None

    if args.model_name == "ucaps":
        net = UCaps3D(**dict_args, class_weight=class_weight)
    elif args.model_name == "segcaps-3d":
        net = SegCaps3D(**dict_args, class_weight=class_weight)
    elif args.model_name == "segcaps-2d":
        net = SegCaps2D(**dict_args, class_weight=class_weight)
    elif args.model_name == "unet":
        net = UNetModule(**dict_args, class_weight=class_weight)

    # set up loggers and checkpoints

    if args.dataset == "iseg2017":
        tb_logger = TensorBoardLogger(save_dir=args.log_dir, name=f"{args.model_name}_{args.dataset}", log_graph=True)
    else:
        tb_logger = TensorBoardLogger(
            save_dir=args.log_dir, name=f"{args.model_name}_{args.dataset}_{args.fold}", log_graph=True
        )
    checkpoint_callback = ModelCheckpoint(
        filename="{epoch}-{val_dice:.4f}", monitor="val_dice", save_top_k=2, mode="max", save_last=True
    )
    earlystopping_callback = EarlyStopping(monitor="val_dice", patience=20, mode="max")

    trainer = Trainer.from_argparse_args(
        args,
        benchmark=True,
        logger=tb_logger,
        callbacks=[checkpoint_callback, earlystopping_callback],
        num_sanity_val_steps=1,
        terminate_on_nan=True,
    )

    trainer.fit(net, datamodule=data_module)
    print("Best model path ", checkpoint_callback.best_model_path)
    print("Best val mean dice ", checkpoint_callback.best_model_score.item())
