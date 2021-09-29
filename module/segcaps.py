from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import pytorch_lightning as pl
import torch
from layers import ConvSlimCapsule2D, ConvSlimCapsule3D, DeconvSlimCapsule2D, DeconvSlimCapsule3D, MarginLoss
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks import one_hot
from monai.networks.blocks import Convolution
from monai.transforms import AsDiscrete, Compose, EnsureType
from monai.visualize.img2tensorboard import plot_2d_or_3d_image
from torch import nn


# Pytorch Lightning module
class SegCaps3D(pl.LightningModule):
    def __init__(
        self,
        in_channels=2,
        out_channels=4,
        lr_rate=2e-4,
        rec_loss_weight=0.1,
        class_weight=None,
        sw_batch_size=1,
        cls_loss="CE",
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
        self.rec_loss_weight = self.hparams.rec_loss_weight
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
        elif self.cls_loss == "Margin":
            self.classification_loss = MarginLoss(class_weight=self.class_weight, margin=0.4)
        self.reconstruction_loss = nn.MSELoss(reduction="none")

        self.val_frequency = self.hparams.val_frequency
        self.val_patch_size = self.hparams.val_patch_size
        self.sw_batch_size = self.hparams.sw_batch_size
        self.overlap = self.hparams.overlap

        # Building model
        self.feature_extractor = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        Convolution(
                            dimensions=3,
                            in_channels=self.in_channels,
                            out_channels=16,
                            kernel_size=5,
                            strides=1,
                            padding=2,
                            bias=True,
                            conv_only=True,
                            act="RELU",
                        ),
                    )
                ]
            )
        )

        self._build_encoder()
        self._build_decoder()
        self._build_reconstruct_branch()

        # For validation
        self.post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=True, n_classes=self.out_channels)])
        self.post_label = Compose([EnsureType(), AsDiscrete(to_onehot=True, n_classes=self.out_channels)])

        self.dice_metric = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False)

        self.example_input_array = torch.rand(1, self.in_channels, 32, 32, 32)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("SegCaps3D")
        # Architecture params
        parser.add_argument("--in_channels", type=int, default=2)
        parser.add_argument("--out_channels", type=int, default=4)

        # Validation params
        parser.add_argument("--val_patch_size", nargs="+", type=int, default=[32, 32, 32])
        parser.add_argument("--val_frequency", type=int, default=100)
        parser.add_argument("--sw_batch_size", type=int, default=1)
        parser.add_argument("--overlap", type=float, default=0.75)

        # Loss params
        parser.add_argument("--rec_loss_weight", type=float, default=1e-1)
        parser.add_argument("--cls_loss", type=str, default="CE")

        # Optimizer params
        parser.add_argument("--lr_rate", type=float, default=2e-4)
        parser.add_argument("--weight_decay", type=float, default=2e-6)
        return parent_parser, parser

    def forward(self, x):
        # Contracting
        x = self.feature_extractor(x)
        conv_cap_1_1 = x.unsqueeze(dim=1)

        x = self.encoder_conv_caps[0](conv_cap_1_1)
        conv_cap_2_1 = self.encoder_conv_caps[1](x)

        x = self.encoder_conv_caps[2](conv_cap_2_1)
        conv_cap_3_1 = self.encoder_conv_caps[3](x)

        x = self.encoder_conv_caps[4](conv_cap_3_1)
        conv_cap_4_1 = self.encoder_conv_caps[5](x)

        # Expanding
        x = self.decoder_conv_caps[0](conv_cap_4_1)
        x = torch.cat((x, conv_cap_3_1), dim=1)
        x = self.decoder_conv_caps[1](x)
        x = self.decoder_conv_caps[2](x)
        x = torch.cat((x, conv_cap_2_1), dim=1)
        x = self.decoder_conv_caps[3](x)
        x = self.decoder_conv_caps[4](x)
        x = torch.cat((x, conv_cap_1_1), dim=1)

        x = self.decoder_conv_caps[5](x)

        logits = torch.linalg.norm(x, dim=2)

        return logits

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]

        # Contracting
        x = self.feature_extractor(images)
        conv_cap_1_1 = x.unsqueeze(dim=1)

        x = self.encoder_conv_caps[0](conv_cap_1_1)
        conv_cap_2_1 = self.encoder_conv_caps[1](x)

        x = self.encoder_conv_caps[2](conv_cap_2_1)
        conv_cap_3_1 = self.encoder_conv_caps[3](x)

        x = self.encoder_conv_caps[4](conv_cap_3_1)
        conv_cap_4_1 = self.encoder_conv_caps[5](x)

        # Expanding
        x = self.decoder_conv_caps[0](conv_cap_4_1)
        x = torch.cat((x, conv_cap_3_1), dim=1)
        x = self.decoder_conv_caps[1](x)
        x = self.decoder_conv_caps[2](x)
        x = torch.cat((x, conv_cap_2_1), dim=1)
        x = self.decoder_conv_caps[3](x)
        x = self.decoder_conv_caps[4](x)
        x = torch.cat((x, conv_cap_1_1), dim=1)

        x = self.decoder_conv_caps[5](x)

        logits = torch.linalg.norm(x, dim=2)

        # Reconstructing
        x_shape = x.size()
        masked_x = x * one_hot(labels, self.out_channels)[:, :, None, :, :, :]
        masked_x = masked_x.reshape(x_shape[0], -1, x_shape[-3], x_shape[-2], x_shape[-1])
        reconstructions = self.reconstruct_branch(masked_x)

        # Calculating losses
        loss, cls_loss, rec_loss = self.losses(images, labels, logits, reconstructions)
        self.log(f"{self.cls_loss}_loss", cls_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("reconstruction_loss", rec_loss, on_step=False, on_epoch=True, sync_dist=True)

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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_rate, weight_decay=self.weight_decay)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", factor=0.1, patience=5),
            "monitor": "val_dice",
            "frequency": self.val_frequency,
        }
        return [optimizer], [scheduler]

    def losses(self, volumes, labels, pred, reconstructions):
        mask = torch.gt(labels, 0)
        rec_loss = torch.sum(self.reconstruction_loss(volumes * mask, reconstructions * mask), dim=(1, 2, 3, 4)) / (
            torch.sum(mask, dim=(1, 2, 3, 4)) + 1e-8
        )
        rec_loss = torch.mean(rec_loss)

        cls_loss = self.classification_loss(pred, labels)

        return (
            cls_loss + self.rec_loss_weight * rec_loss,
            cls_loss,
            rec_loss,
        )

    def _build_encoder(self):
        self.encoder_conv_caps = nn.ModuleList()
        self.encoder_kernel_size = 5
        self.encoder_output_dim = [2, 4, 4, 8, 8, 8]
        self.encoder_output_atoms = [16, 16, 32, 32, 64, 32]

        for i in range(len(self.encoder_output_dim)):
            if i == 0:
                input_dim = 1
                input_atoms = 16
            else:
                input_dim = self.encoder_output_dim[i - 1]
                input_atoms = self.encoder_output_atoms[i - 1]

            stride = 2 if i % 2 == 0 else 1

            self.encoder_conv_caps.append(
                ConvSlimCapsule3D(
                    kernel_size=self.encoder_kernel_size,
                    input_dim=input_dim,
                    output_dim=self.encoder_output_dim[i],
                    input_atoms=input_atoms,
                    output_atoms=self.encoder_output_atoms[i],
                    stride=stride,
                    padding=2,
                    dilation=1,
                    num_routing=3,
                    share_weight=True,
                )
            )

    def _build_decoder(self):
        self.decoder_conv_caps = nn.ModuleList()
        self.decoder_input_dim = [8, 16, 4, 8, 4, 3]
        self.decoder_output_dim = [8, 4, 4, 4, 2, self.out_channels]
        self.decoder_output_atoms = [32, 32, 16, 16, 16, 16]

        for i in range(len(self.decoder_output_dim)):
            if i == 0:
                input_atoms = self.encoder_output_atoms[-1]
            else:
                input_atoms = self.decoder_output_atoms[i - 1]

            if i % 2 == 0:
                self.decoder_conv_caps.append(
                    DeconvSlimCapsule3D(
                        kernel_size=4,
                        input_dim=self.decoder_input_dim[i],
                        output_dim=self.decoder_output_dim[i],
                        input_atoms=input_atoms,
                        output_atoms=self.decoder_output_atoms[i],
                        stride=2,
                        padding=1,
                        num_routing=3,
                        share_weight=True,
                    )
                )
            else:
                self.decoder_conv_caps.append(
                    ConvSlimCapsule3D(
                        kernel_size=5,
                        input_dim=self.decoder_input_dim[i],
                        output_dim=self.decoder_output_dim[i],
                        input_atoms=input_atoms,
                        output_atoms=self.decoder_output_atoms[i],
                        stride=1,
                        padding=2,
                        num_routing=3,
                        share_weight=True,
                    )
                )

    def _build_reconstruct_branch(self):
        self.reconstruct_branch = nn.Sequential(
            nn.Conv3d(self.decoder_output_atoms[-1] * self.out_channels, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, self.in_channels, 1),
            nn.Sigmoid(),
        )


class SegCaps2D(pl.LightningModule):
    def __init__(
        self,
        input_dim=3,
        in_channels=2,
        out_channels=4,
        lr_rate=2e-4,
        rec_loss_weight=0.1,
        class_weight=None,
        sw_batch_size=128,
        cls_loss="CE",
        val_patch_size=(-1, -1, 1),
        overlap=0.75,
        val_frequency=100,
        weight_decay=2e-6,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.input_dim = self.hparams.input_dim
        self.in_channels = self.hparams.in_channels
        self.out_channels = self.hparams.out_channels

        self.lr_rate = self.hparams.lr_rate
        self.weight_decay = self.hparams.weight_decay

        self.cls_loss = self.hparams.cls_loss
        self.rec_loss_weight = self.hparams.rec_loss_weight
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
        elif self.cls_loss == "Margin":
            self.classification_loss = MarginLoss(class_weight=self.class_weight, margin=0.4)
        self.reconstruction_loss = nn.MSELoss(reduction="none")

        self.val_frequency = self.hparams.val_frequency
        self.val_patch_size = self.hparams.val_patch_size
        self.sw_batch_size = self.hparams.sw_batch_size
        self.overlap = self.hparams.overlap

        # Building model
        self.feature_extractor = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        Convolution(
                            dimensions=2,
                            in_channels=self.in_channels,
                            out_channels=16,
                            kernel_size=5,
                            strides=1,
                            padding=2,
                            bias=True,
                            conv_only=True,
                            act="RELU",
                        ),
                    )
                ]
            )
        )

        self._build_encoder()
        self._build_decoder()
        self._build_reconstruct_branch()

        # For validation
        self.post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=True, n_classes=self.out_channels)])
        self.post_label = Compose([EnsureType(), AsDiscrete(to_onehot=True, n_classes=self.out_channels)])

        self.dice_metric = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False)

        if self.input_dim == 3:
            self.example_input_array = torch.rand(1, self.in_channels, 32, 32, 1)
        else:
            self.example_input_array = torch.rand(1, self.in_channels, 32, 32)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("SegCaps2D")
        # Architecture params
        parser.add_argument("--input_dim", type=int, default=3)
        parser.add_argument("--in_channels", type=int, default=2)
        parser.add_argument("--out_channels", type=int, default=4)

        # Validation params
        parser.add_argument("--val_patch_size", nargs="+", type=int, default=[-1, -1, 1])
        parser.add_argument("--val_frequency", type=int, default=100)
        parser.add_argument("--sw_batch_size", type=int, default=1)
        parser.add_argument("--overlap", type=float, default=0.75)

        # Loss params
        parser.add_argument("--rec_loss_weight", type=float, default=1e-1)
        parser.add_argument("--cls_loss", type=str, default="CE")

        # Optimizer params
        parser.add_argument("--lr_rate", type=float, default=2e-4)
        parser.add_argument("--weight_decay", type=float, default=2e-6)
        return parent_parser, parser

    def forward(self, x):
        if self.input_dim == 3:
            x = x.squeeze(dim=-1)

        # Contracting
        x = self.feature_extractor(x)
        conv_cap_1_1 = x.unsqueeze(dim=1)

        x = self.encoder_conv_caps[0](conv_cap_1_1)
        conv_cap_2_1 = self.encoder_conv_caps[1](x)

        x = self.encoder_conv_caps[2](conv_cap_2_1)
        conv_cap_3_1 = self.encoder_conv_caps[3](x)

        x = self.encoder_conv_caps[4](conv_cap_3_1)
        conv_cap_4_1 = self.encoder_conv_caps[5](x)

        # Expanding
        x = self.decoder_conv_caps[0](conv_cap_4_1)
        x = torch.cat((x, conv_cap_3_1), dim=1)
        x = self.decoder_conv_caps[1](x)
        x = self.decoder_conv_caps[2](x)
        x = torch.cat((x, conv_cap_2_1), dim=1)
        x = self.decoder_conv_caps[3](x)
        x = self.decoder_conv_caps[4](x)
        x = torch.cat((x, conv_cap_1_1), dim=1)

        x = self.decoder_conv_caps[5](x)

        logits = torch.linalg.norm(x, dim=2)

        if self.input_dim == 3:
            logits = logits.unsqueeze(dim=-1)

        return logits

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]

        if self.input_dim == 3:
            images = images.squeeze(dim=-1)
            labels = labels.squeeze(dim=-1)

        # Contracting
        x = self.feature_extractor(images)
        conv_cap_1_1 = x.unsqueeze(dim=1)

        x = self.encoder_conv_caps[0](conv_cap_1_1)
        conv_cap_2_1 = self.encoder_conv_caps[1](x)

        x = self.encoder_conv_caps[2](conv_cap_2_1)
        conv_cap_3_1 = self.encoder_conv_caps[3](x)

        x = self.encoder_conv_caps[4](conv_cap_3_1)
        conv_cap_4_1 = self.encoder_conv_caps[5](x)

        # Expanding
        x = self.decoder_conv_caps[0](conv_cap_4_1)
        x = torch.cat((x, conv_cap_3_1), dim=1)
        x = self.decoder_conv_caps[1](x)
        x = self.decoder_conv_caps[2](x)
        x = torch.cat((x, conv_cap_2_1), dim=1)
        x = self.decoder_conv_caps[3](x)
        x = self.decoder_conv_caps[4](x)
        x = torch.cat((x, conv_cap_1_1), dim=1)

        x = self.decoder_conv_caps[5](x)

        logits = torch.linalg.norm(x, dim=2)

        # Reconstructing
        x_shape = x.size()
        masked_x = x * one_hot(labels, self.out_channels)[:, :, None, :, :]
        masked_x = masked_x.reshape(x_shape[0], -1, x_shape[-2], x_shape[-1])
        reconstructions = self.reconstruct_branch(masked_x)

        # Calculating losses
        loss, cls_loss, rec_loss = self.losses(images, labels, logits, reconstructions)
        self.log(f"{self.cls_loss}_loss", cls_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("reconstruction_loss", rec_loss, on_step=False, on_epoch=True, sync_dist=True)

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

    def losses(self, volumes, labels, pred, reconstructions):
        mask = torch.gt(labels, 0)
        rec_loss = torch.sum(self.reconstruction_loss(volumes * mask, reconstructions * mask), dim=(1, 2, 3)) / (
            torch.sum(mask, dim=(1, 2, 3)) + 1e-8
        )
        rec_loss = torch.mean(rec_loss)

        cls_loss = self.classification_loss(pred, labels)

        return (
            cls_loss + self.rec_loss_weight * rec_loss,
            cls_loss,
            rec_loss,
        )

    def _build_encoder(self):
        self.encoder_conv_caps = nn.ModuleList()
        self.encoder_kernel_size = 5
        self.encoder_output_dim = [2, 4, 4, 8, 8, 8]
        self.encoder_output_atoms = [16, 16, 32, 32, 64, 32]

        for i in range(len(self.encoder_output_dim)):
            if i == 0:
                input_dim = 1
                input_atoms = 16
            else:
                input_dim = self.encoder_output_dim[i - 1]
                input_atoms = self.encoder_output_atoms[i - 1]

            stride = 2 if i % 2 == 0 else 1

            self.encoder_conv_caps.append(
                ConvSlimCapsule2D(
                    kernel_size=self.encoder_kernel_size,
                    input_dim=input_dim,
                    output_dim=self.encoder_output_dim[i],
                    input_atoms=input_atoms,
                    output_atoms=self.encoder_output_atoms[i],
                    stride=stride,
                    padding=2,
                    dilation=1,
                    num_routing=3,
                    share_weight=True,
                )
            )

    def _build_decoder(self):
        self.decoder_conv_caps = nn.ModuleList()
        self.decoder_input_dim = [8, 16, 4, 8, 4, 3]
        self.decoder_output_dim = [8, 4, 4, 4, 2, self.out_channels]
        self.decoder_output_atoms = [32, 32, 16, 16, 16, 16]

        for i in range(len(self.decoder_output_dim)):
            if i == 0:
                input_atoms = self.encoder_output_atoms[-1]
            else:
                input_atoms = self.decoder_output_atoms[i - 1]

            if i % 2 == 0:
                self.decoder_conv_caps.append(
                    DeconvSlimCapsule2D(
                        kernel_size=4,
                        input_dim=self.decoder_input_dim[i],
                        output_dim=self.decoder_output_dim[i],
                        input_atoms=input_atoms,
                        output_atoms=self.decoder_output_atoms[i],
                        stride=2,
                        padding=1,
                        num_routing=3,
                        share_weight=True,
                    )
                )
            else:
                self.decoder_conv_caps.append(
                    ConvSlimCapsule2D(
                        kernel_size=5,
                        input_dim=self.decoder_input_dim[i],
                        output_dim=self.decoder_output_dim[i],
                        input_atoms=input_atoms,
                        output_atoms=self.decoder_output_atoms[i],
                        stride=1,
                        padding=2,
                        num_routing=3,
                        share_weight=True,
                    )
                )

    def _build_reconstruct_branch(self):
        self.reconstruct_branch = nn.Sequential(
            nn.Conv2d(self.decoder_output_atoms[-1] * self.out_channels, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self.in_channels, 1),
            nn.Sigmoid(),
        )
