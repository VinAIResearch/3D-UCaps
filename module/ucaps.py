from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from layers import ConvSlimCapsule3D, MarginLoss
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks import one_hot
from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv
from monai.transforms import AsDiscrete, Compose, EnsureType
from monai.visualize.img2tensorboard import plot_2d_or_3d_image
from torch import nn


# Pytorch Lightning module
class UCaps3D(pl.LightningModule):
    def __init__(
        self,
        in_channels=2,
        out_channels=4,
        lr_rate=2e-4,
        rec_loss_weight=0.1,
        margin_loss_weight=1.0,
        class_weight=None,
        share_weight=False,
        sw_batch_size=128,
        cls_loss="CE",
        val_patch_size=(32, 32, 32),
        overlap=0.75,
        connection="skip",
        val_frequency=100,
        weight_decay=2e-6,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.in_channels = self.hparams.in_channels
        self.out_channels = self.hparams.out_channels
        self.share_weight = self.hparams.share_weight
        self.connection = self.hparams.connection

        self.lr_rate = self.hparams.lr_rate
        self.weight_decay = self.hparams.weight_decay

        self.cls_loss = self.hparams.cls_loss
        self.margin_loss_weight = self.hparams.margin_loss_weight
        self.rec_loss_weight = self.hparams.rec_loss_weight
        self.class_weight = self.hparams.class_weight

        # Defining losses
        self.classification_loss1 = MarginLoss(class_weight=self.class_weight, margin=0.2)

        if self.cls_loss == "DiceCE":
            self.classification_loss2 = DiceCELoss(softmax=True, to_onehot_y=True, ce_weight=self.class_weight)
        elif self.cls_loss == "CE":
            self.classification_loss2 = DiceCELoss(
                softmax=True, to_onehot_y=True, ce_weight=self.class_weight, lambda_dice=0.0
            )
        elif self.cls_loss == "Dice":
            self.classification_loss2 = DiceCELoss(softmax=True, to_onehot_y=True, lambda_ce=0.0)
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
                            bias=False,
                        ),
                    ),
                    (
                        "conv2",
                        Convolution(
                            dimensions=3,
                            in_channels=16,
                            out_channels=32,
                            kernel_size=5,
                            strides=1,
                            dilation=2,
                            padding=4,
                            bias=False,
                        ),
                    ),
                    (
                        "conv3",
                        Convolution(
                            dimensions=3,
                            in_channels=32,
                            out_channels=64,
                            kernel_size=5,
                            strides=1,
                            padding=4,
                            dilation=2,
                            bias=False,
                            act="tanh",
                        ),
                    ),
                ]
            )
        )

        self.primary_caps = ConvSlimCapsule3D(
            kernel_size=3,
            input_dim=1,
            output_dim=16,
            input_atoms=64,
            output_atoms=4,
            stride=1,
            padding=1,
            num_routing=1,
            share_weight=self.share_weight,
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
        parser = parent_parser.add_argument_group("UCaps3D")
        # Architecture params
        parser.add_argument("--in_channels", type=int, default=2)
        parser.add_argument("--out_channels", type=int, default=4)
        parser.add_argument("--share_weight", type=int, default=1)
        parser.add_argument("--connection", type=str, default="skip")

        # Validation params
        parser.add_argument("--val_patch_size", nargs="+", type=int, default=[32, 32, 32])
        parser.add_argument("--val_frequency", type=int, default=100)
        parser.add_argument("--sw_batch_size", type=int, default=1)
        parser.add_argument("--overlap", type=float, default=0.75)

        # Loss params
        parser.add_argument("--margin_loss_weight", type=float, default=0.1)
        parser.add_argument("--rec_loss_weight", type=float, default=1e-1)
        parser.add_argument("--cls_loss", type=str, default="CE")

        # Optimizer params
        parser.add_argument("--lr_rate", type=float, default=2e-4)
        parser.add_argument("--weight_decay", type=float, default=2e-6)
        return parent_parser, parser

    def forward(self, x):
        # Contracting
        x = self.feature_extractor(x)
        x = x.unsqueeze(dim=1)
        conv_cap_1_1 = self.primary_caps(x)

        x = self.encoder_conv_caps[0](conv_cap_1_1)
        conv_cap_2_1 = self.encoder_conv_caps[1](x)

        x = self.encoder_conv_caps[2](conv_cap_2_1)
        conv_cap_3_1 = self.encoder_conv_caps[3](x)

        x = self.encoder_conv_caps[4](conv_cap_3_1)
        conv_cap_4_1 = self.encoder_conv_caps[5](x)

        shape = conv_cap_4_1.size()
        conv_cap_4_1 = conv_cap_4_1.view(shape[0], -1, shape[-3], shape[-2], shape[-1])
        shape = conv_cap_3_1.size()
        conv_cap_3_1 = conv_cap_3_1.view(shape[0], -1, shape[-3], shape[-2], shape[-1])
        shape = conv_cap_2_1.size()
        conv_cap_2_1 = conv_cap_2_1.view(shape[0], -1, shape[-3], shape[-2], shape[-1])
        shape = conv_cap_1_1.size()
        conv_cap_1_1 = conv_cap_1_1.view(shape[0], -1, shape[-3], shape[-2], shape[-1])

        # Expanding
        if self.connection == "skip":
            x = self.decoder_conv[0](conv_cap_4_1)
            x = torch.cat((x, conv_cap_3_1), dim=1)
            x = self.decoder_conv[1](x)
            x = self.decoder_conv[2](x)
            x = torch.cat((x, conv_cap_2_1), dim=1)
            x = self.decoder_conv[3](x)
            x = self.decoder_conv[4](x)
            x = torch.cat((x, conv_cap_1_1), dim=1)

        logits = self.decoder_conv[5](x)

        return logits

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]

        # Contracting
        x = self.feature_extractor(images)
        x = x.unsqueeze(dim=1)
        conv_cap_1_1 = self.primary_caps(x)

        x = self.encoder_conv_caps[0](conv_cap_1_1)
        conv_cap_2_1 = self.encoder_conv_caps[1](x)

        x = self.encoder_conv_caps[2](conv_cap_2_1)
        conv_cap_3_1 = self.encoder_conv_caps[3](x)

        x = self.encoder_conv_caps[4](conv_cap_3_1)
        conv_cap_4_1 = self.encoder_conv_caps[5](x)

        # Downsampled predictions
        norm = torch.linalg.norm(conv_cap_4_1, dim=2)

        shape = conv_cap_4_1.size()
        conv_cap_4_1 = conv_cap_4_1.view(shape[0], -1, shape[-3], shape[-2], shape[-1])
        shape = conv_cap_3_1.size()
        conv_cap_3_1 = conv_cap_3_1.view(shape[0], -1, shape[-3], shape[-2], shape[-1])
        shape = conv_cap_2_1.size()
        conv_cap_2_1 = conv_cap_2_1.view(shape[0], -1, shape[-3], shape[-2], shape[-1])
        shape = conv_cap_1_1.size()
        conv_cap_1_1 = conv_cap_1_1.view(shape[0], -1, shape[-3], shape[-2], shape[-1])

        # Expanding
        if self.connection == "skip":
            x = self.decoder_conv[0](conv_cap_4_1)
            x = torch.cat((x, conv_cap_3_1), dim=1)
            x = self.decoder_conv[1](x)
            x = self.decoder_conv[2](x)
            x = torch.cat((x, conv_cap_2_1), dim=1)
            x = self.decoder_conv[3](x)
            x = self.decoder_conv[4](x)
            x = torch.cat((x, conv_cap_1_1), dim=1)

        logits = self.decoder_conv[5](x)

        # Reconstructing
        reconstructions = self.reconstruct_branch(x)

        # Calculating losses
        loss, cls_loss, rec_loss = self.losses(images, labels, norm, logits, reconstructions)

        self.log("margin_loss", cls_loss[0], on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{self.cls_loss}_loss", cls_loss[1], on_step=False, on_epoch=True, sync_dist=True)
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

    def losses(self, volumes, labels, norm, pred, reconstructions):
        mask = torch.gt(labels, 0)
        rec_loss = torch.sum(self.reconstruction_loss(volumes * mask, reconstructions * mask), dim=(1, 2, 3, 4)) / (
            torch.sum(mask, dim=(1, 2, 3, 4)) + 1e-8
        )
        rec_loss = torch.mean(rec_loss)

        downsample_labels = F.interpolate(
            one_hot(labels, self.out_channels), scale_factor=0.125, mode="trilinear", align_corners=False
        )
        cls_loss1 = self.classification_loss1(norm, downsample_labels)
        cls_loss2 = self.classification_loss2(pred, labels)

        return (
            self.margin_loss_weight * cls_loss1 + cls_loss2 + self.rec_loss_weight * rec_loss,
            [cls_loss1, cls_loss2],
            rec_loss,
        )

    def _build_encoder(self):
        self.encoder_conv_caps = nn.ModuleList()
        self.encoder_kernel_size = 3
        self.encoder_output_dim = [16, 16, 8, 8, 8, self.out_channels]
        self.encoder_output_atoms = [8, 8, 16, 16, 32, 64]

        for i in range(len(self.encoder_output_dim)):
            if i == 0:
                input_dim = self.primary_caps.output_dim
                input_atoms = self.primary_caps.output_atoms
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
                    padding=1,
                    dilation=1,
                    num_routing=3,
                    share_weight=self.share_weight,
                )
            )

    def _build_decoder(self):
        self.decoder_conv = nn.ModuleList()
        if self.connection == "skip":
            self.decoder_in_channels = [self.out_channels * self.encoder_output_atoms[-1], 384, 128, 256, 64, 128]
            self.decoder_out_channels = [256, 128, 128, 64, 64, self.out_channels]

        for i in range(6):
            if i == 5:
                self.decoder_conv.append(
                    Conv["conv", 3](self.decoder_in_channels[i], self.decoder_out_channels[i], kernel_size=1)
                )
            elif i % 2 == 0:
                self.decoder_conv.append(
                    UpSample(
                        dimensions=3,
                        in_channels=self.decoder_in_channels[i],
                        out_channels=self.decoder_out_channels[i],
                        scale_factor=2,
                    )
                )
            else:
                self.decoder_conv.append(
                    Convolution(
                        dimensions=3,
                        kernel_size=3,
                        in_channels=self.decoder_in_channels[i],
                        out_channels=self.decoder_out_channels[i],
                        strides=1,
                        padding=1,
                        bias=False,
                    )
                )

    def _build_reconstruct_branch(self):
        self.reconstruct_branch = nn.Sequential(
            nn.Conv3d(self.decoder_in_channels[-1], 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, self.in_channels, 1),
            nn.Sigmoid(),
        )
