from __future__ import absolute_import, division, print_function

import pytorch_lightning as pl
import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import BasicUNet
from monai.transforms import AsDiscrete, Compose, EnsureType
from monai.visualize.img2tensorboard import plot_2d_or_3d_image


class UNetModule(pl.LightningModule):
    def __init__(
        self,
        in_channels=2,
        out_channels=4,
        lr_rate=2e-4,
        class_weight=None,
        sw_batch_size=128,
        cls_loss="DiceCE",
        val_patch_size=(32, 32, 32),
        overlap=0.75,
        val_frequency=100,
        weight_decay=2e-6,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.in_channels = self.hparams.in_channels
        self.out_channels = self.hparams.out_channels

        self.lr_rate = self.hparams.lr_rate
        self.weight_decay = self.hparams.weight_decay

        self.cls_loss = self.hparams.cls_loss
        self.class_weight = self.hparams.class_weight

        # Defining losses
        if self.cls_loss == "DiceCE":
            self.classification_loss = DiceCELoss(softmax=True, to_onehot_y=True, ce_weight=self.class_weight)
        elif self.cls_loss == "CE":
            self.classification_loss = DiceCELoss(
                softmax=True, to_onehot_y=True, ce_weight=self.class_weight, lambda_dice=0.0
            )
        elif self.cls_loss == "Dice":
            self.classification_loss = DiceCELoss(softmax=True, to_onehot_y=True, lambda_ce=0.0)

        self.val_frequency = self.hparams.val_frequency
        self.val_patch_size = self.hparams.val_patch_size
        self.sw_batch_size = self.hparams.sw_batch_size
        self.overlap = self.hparams.overlap

        # Building model
        self.model = BasicUNet(in_channels=self.in_channels, out_channels=self.out_channels)

        # For validation
        self.post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=True, n_classes=self.out_channels)])
        self.post_label = Compose([EnsureType(), AsDiscrete(to_onehot=True, n_classes=self.out_channels)])

        self.dice_metric = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False)

        self.example_input_array = torch.rand(1, self.in_channels, 32, 32, 32)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("UNet3D")
        # Architecture params
        parser.add_argument("--in_channels", type=int, default=2)
        parser.add_argument("--out_channels", type=int, default=4)

        # Validation params
        parser.add_argument("--val_patch_size", nargs="+", type=int, default=[32, 32, 32])
        parser.add_argument("--val_frequency", type=int, default=100)
        parser.add_argument("--sw_batch_size", type=int, default=128)
        parser.add_argument("--overlap", type=float, default=0.75)

        # Loss params
        parser.add_argument("--cls_loss", type=str, default="DiceCE")

        # Optimizer params
        parser.add_argument("--lr_rate", type=float, default=2e-4)
        parser.add_argument("--weight_decay", type=float, default=2e-6)
        return parent_parser, parser

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]

        logits = self.model(images)

        loss = self.classification_loss(logits, labels)

        self.log(f"{self.cls_loss}_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]

        val_outputs = sliding_window_inference(
            images,
            roi_size=self.val_patch_size,
            sw_batch_size=self.sw_batch_size,
            predictor=self.forward,
            overlap=self.overlap,
        )

        # Visualize to tensorboard
        if self.global_rank == 0 and batch_idx == 0:
            plot_2d_or_3d_image(
                images,
                step=self.global_step,
                writer=self.logger.experiment,
                max_channels=self.in_channels,
                tag="Input Image",
            )
            plot_2d_or_3d_image(labels * 20, step=self.global_step, writer=self.logger.experiment, tag="Label")
            plot_2d_or_3d_image(
                torch.argmax(val_outputs, dim=1, keepdim=True) * 20,
                step=self.global_step,
                writer=self.logger.experiment,
                tag="Prediction",
            )

        val_outputs = [self.post_pred(val_output) for val_output in decollate_batch(val_outputs)]
        labels = [self.post_label(label) for label in decollate_batch(labels)]
        self.dice_metric(y_pred=val_outputs, y=labels)

    def validation_epoch_end(self, outputs):
        dice_scores = self.dice_metric.aggregate()
        mean_val_dice = torch.mean(dice_scores)
        self.log("val_dice", mean_val_dice, sync_dist=True)
        for i, dice_score in enumerate(dice_scores):
            self.log(f"val_dice_class {i + 1}", dice_score, sync_dist=True)
        self.dice_metric.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        images = batch["image"]
        outputs = sliding_window_inference(
            images,
            roi_size=self.val_patch_size,
            sw_batch_size=self.sw_batch_size,
            predictor=self.forward,
            overlap=self.overlap,
        )
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_rate, weight_decay=self.weight_decay)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", factor=0.1, patience=5),
            "monitor": "val_dice",
            "frequency": self.val_frequency,
        }
        return [optimizer], [scheduler]
