"""
import argparse

import SimpleITK as sitk
import torch
from common import *
from medpy.metric.binary import *
from monai.data import NiftiSaver
from monai.inferers import sliding_window_inference
from monai.metrics import compute_meandice
from monai.transforms import (
    AddChanneld,
    AsChannelFirstd,
    AsDiscrete,
    Compose,
    MapLabelValue,
    NormalizeIntensityd,
    Rotated,
    ToTensord,
)


#############################
# Read Nii/hdr file using stk
#############################
def read_med_image(file_path, dtype):
    img_stk = sitk.ReadImage(file_path)
    img_np = sitk.GetArrayFromImage(img_stk)
    img_np = img_np.astype(dtype)
    return img_np, img_stk


def convert_label(label_img):
    label_processed = np.zeros(label_img.shape[0:]).astype(np.uint8)
    for i in range(label_img.shape[2]):
        label_slice = label_img[:, :, i]
        label_slice[label_slice == 10] = 1
        label_slice[label_slice == 150] = 2
        label_slice[label_slice == 250] = 3
        label_processed[:, :, i] = label_slice
    return label_processed


net = DenseNet(num_init_features=32, growth_rate=16, block_config=(4, 4, 4, 4), num_classes=4)

parser = argparse.ArgumentParser()
parser.add_argument("--test_path", type=str, default="/home/ubuntu/domainA_val")
parser.add_argument("--gpu", type=int, default=0, help="Gpu index")
parser.add_argument("--save_image", type=int, default=0, help="Save image or not")

# Validation config
val_parser = parser.add_argument_group("Validation config")
val_parser.add_argument("--output_dir", type=str, default="/home/ubuntu")
val_parser.add_argument(
    "--patch_size", nargs="+", type=int, default=[32, 32, 32], help="Patch size using to validate model"
)
val_parser.add_argument("--mode", type=str, default="constant", help="constant or gaussian")
val_parser.add_argument("--sigma_scale", type=float, default=0.25)
val_parser.add_argument("--sw_batch_size", type=int, default=1)
val_parser.add_argument("--overlap", type=float, default=0.25)
val_parser.add_argument("--rotate_angle", type=int, default=0)
val_parser.add_argument("--axis", type=str, default="z", help="z/y/x or all")

if __name__ == "__main__":
    args = parser.parse_args()
    dict_args = vars(args)

    print("Validation config:")
    for a in val_parser._group_actions:
        print("\t{}:\t{}".format(a.dest, dict_args[a.dest]))

    # -----------------------Testing-------------------------------------
    # -----------------------Load the checkpoint (weights)---------------
    checkpoint = "checkpoints/20000_model_3d_denseseg_v1.pth"
    print("Checkpoint: ", checkpoint)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    saved_state_dict = torch.load(checkpoint, map_location="cpu")
    net.load_state_dict(saved_state_dict)
    net.to(device)
    net.eval()
    # -----------------------Load testing data----------------------------
    test_path = args.test_path
    subject_id = 9
    subject_name = "subject-%d-" % subject_id

    f_T1 = os.path.join(test_path, subject_name + "T1.hdr")
    f_T2 = os.path.join(test_path, subject_name + "T2.hdr")
    f_l = os.path.join(test_path, subject_name + "label.hdr")
    inputs_T1, img_T1_itk = read_med_image(f_T1, dtype=np.float32)
    inputs_T2, img_T2_itk = read_med_image(f_T2, dtype=np.float32)
    label, label_img_itk = read_med_image(f_l, dtype=np.uint8)
    label = label.astype(np.uint8)
    label = convert_label(label)

    mask = inputs_T1 > 0
    mask = mask.astype(np.bool)
    # # ======================normalize to 0 mean and 1 variance====
    # # Normalization
    subtrahend = [inputs_T1[mask].mean(), inputs_T2[mask].mean()]
    divisor = [inputs_T1[mask].std(), inputs_T2[mask].std()]
    normalize_transform = NormalizeIntensityd(
        keys=["image"], subtrahend=subtrahend, divisor=divisor, channel_wise=True
    )

    inputs_T1 = inputs_T1[:, :, :, None]
    inputs_T2 = inputs_T2[:, :, :, None]
    inputs = np.concatenate((inputs_T1, inputs_T2), axis=3)

    rotate_angle = args.rotate_angle * np.pi / 180
    if args.axis == "z":
        rotate_transform = Rotated(
            keys=["image", "label"], angle=(rotate_angle, 0, 0), keep_size=False, mode=("bilinear", "nearest")
        )
    elif args.axis == "y":
        rotate_transform = Rotated(
            keys=["image", "label"], angle=(0, rotate_angle, 0), keep_size=False, mode=("bilinear", "nearest")
        )
    elif args.axis == "x":
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
            AddChanneld(keys=["label"]),
            AsChannelFirstd(keys=["image"]),
            rotate_transform,
            normalize_transform,
            ToTensord(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
        ]
    )
    sample = val_transforms({"image": inputs, "label": label})

    map_label = MapLabelValue(target_labels=[0, 10, 150, 250], orig_labels=[0, 1, 2, 3], dtype=np.uint8)

    image_saver = NiftiSaver(
        output_dir=args.output_dir,
        output_postfix=f"{args.rotate_angle}_degree_{args.axis}_axis",
        resample=False,
        data_root_dir=args.test_path[:-13],
    )
    label_saver = NiftiSaver(
        output_dir=args.output_dir,
        output_postfix=f"{args.rotate_angle}_degree_{args.axis}_axis",
        resample=False,
        data_root_dir=args.test_path[:-13],
        output_dtype=np.uint8,
    )
    pred_saver = NiftiSaver(
        output_dir=args.output_dir,
        output_postfix=f"{args.rotate_angle}_degree_{args.axis}_axis_prediction",
        resample=False,
        data_root_dir=args.test_path[:-13],
        output_dtype=np.uint8,
    )

    image_saver.save_batch(sample["image"].permute(0, 1, 4, 3, 2), meta_data={"filename_or_obj": [f_T1]})
    label_saver.save_batch(map_label(sample["label"].permute(0, 1, 4, 3, 2)), meta_data={"filename_or_obj": [f_l]})

    inputs, label = sample["image"], sample["label"]
    image = inputs.permute(0, 1, 2, 4, 3)
    label = label.permute(0, 1, 2, 4, 3)
    image = image.to(device)
    label = label.to(device)

    print(image.size())

    with torch.no_grad():
        outputs = sliding_window_inference(
            image,
            roi_size=args.patch_size,
            sw_batch_size=args.sw_batch_size,
            predictor=net.forward,
            overlap=args.overlap,
            sigma_scale=args.sigma_scale,
            mode=args.mode,
            device=torch.device("cpu"),
        )

    post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=4)
    save_pred = AsDiscrete(argmax=True, to_onehot=False, n_classes=4)
    post_label = AsDiscrete(to_onehot=True, n_classes=4)

    pred_saver.save_batch(map_label(save_pred(outputs.permute(0, 1, 3, 4, 2))), meta_data={"filename_or_obj": [f_l]})

    dice_value = compute_meandice(y_pred=post_pred(outputs), y=post_label(label).cpu(), include_background=False)
    dice_scores = dice_value.mean(dim=0).cpu().numpy()
    mean_dice_score = np.mean(dice_scores)

    print("Validation dice score: %.4f" % mean_dice_score)
    print(
        {
            "Validation Dice Score CSF": dice_scores[0],
            "Validation Dice Score GM": dice_scores[1],
            "Validation Dice Score WM": dice_scores[2],
        }
    )

    print("Finished Evaluation")
"""
