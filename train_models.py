from monai.utils import set_determinism
from monai.data import DataLoader, Dataset, decollate_batch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    ToTensord,
    AsDiscrete,
    SpatialPadd
)
from monai.networks.nets import UNet, UNETR, SwinUNETR
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.visualize import matshow3d
from pathlib import Path
from tqdm import tqdm
from typing import Literal
import torch
import matplotlib.pyplot as plt
import pandas as pd
import logging
import datetime
import argparse
import json
import time


class Trainer:
    def __init__(self, model_name: Literal["unet", "unetr", "swinunetr"]):
        self.model_name = model_name
        self.unet_properties = {
            "img_size": (48, 48, 16),
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 2,
            "channels": (16, 32, 64, 128, 256),
            "strides": (2, 2, 2, 2),
            "num_res_units": 2,
            "norm": Norm.BATCH
        }
        self.unetr_properties = {
            "in_channels": 1,
            "out_channels": 2,
            "img_size": (48, 48, 16),
            "norm_name": 'batch'
        }
        self.swinunetr_properties = {
            "in_channels": 1,
            "out_channels": 2,
            "img_size": (64, 64, 32),
            "norm_name": 'batch'
        }
        self.post_prediction = Compose([
            AsDiscrete(argmax=True, to_onehot=2)
        ])
        self.post_label = Compose([
            AsDiscrete(to_onehot=2)
        ])
        self.verbose=True

    @property
    def unet_name(self):
        return "unet"

    @property
    def unetr_name(self):
        return "unetr"

    @property
    def swinunetr_name(self):
        return "swinunetr"

    @property
    def unet(self):
        return UNet(
            spatial_dims=self.unet_properties["spatial_dims"],
            in_channels=self.unet_properties["in_channels"],
            out_channels=self.unet_properties["out_channels"],
            channels=self.unet_properties["channels"],
            strides=self.unet_properties["strides"],
            num_res_units=self.unet_properties["num_res_units"],
            norm=self.unet_properties["norm"]
        )

    @property
    def unetr(self):
        return UNETR(
            in_channels=self.unetr_properties["in_channels"],
            out_channels=self.unetr_properties["out_channels"],
            img_size=self.unetr_properties["img_size"],
            norm_name=self.unetr_properties["norm_name"]
        )

    @property
    def swinunetr(self):
        return SwinUNETR(
            img_size=self.swinunetr_properties["img_size"],
            in_channels=self.swinunetr_properties["in_channels"],
            out_channels=self.swinunetr_properties["out_channels"],
            norm_name=self.swinunetr_properties["norm_name"]
        )

    def get_tr_transforms(self, padding_spatial_size):
        return Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityd(keys=["image"]),
            ScaleIntensityd(keys=["label"], dtype=torch.uint8),
            SpatialPadd(keys=["image", "label"], spatial_size=padding_spatial_size),
            ToTensord(keys=["image", "label"])
        ])

    def get_val_transforms(self, padding_spatial_size):
        return Compose([
            LoadImaged(keys=["image", "label"], image_only=False),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityd(keys=["image"]),
            ScaleIntensityd(keys=["label"], dtype=torch.uint8),
            SpatialPadd(keys=["image", "label"], spatial_size=padding_spatial_size),
            ToTensord(keys=["image", "label"])
        ])

    def train(self, path_to_images, path_to_masks, path_to_output,
              training_samples_csv=None, val_samples_csv=None,
              max_epochs=100, val_interval=10, batch_size=4,
              num_workers=4, device='cpu'):
        # Check input parameters
        if bool(training_samples_csv) ^ bool(val_samples_csv):
            raise ValueError(
                "both 'training_samples_csv' and 'val_samples_csv' "
                "(or none) have to be specified"
            )
        # Create dataset and dataloaders
        if not training_samples_csv:
            paths_to_images = sorted(list(Path(path_to_images).glob('*.tif')))
            paths_to_masks = sorted(list(Path(path_to_masks).glob('*.tif')))
            tr_data_dicts = [
                {
                    "image": str(path_to_image),
                    "label": str(path_to_label)
                }
                for path_to_image, path_to_label in zip(paths_to_images, paths_to_masks)
            ]
        else:
            tr_df = pd.read_csv(training_samples_csv)
            tr_data_dicts = [
                {
                    "image": str(Path(path_to_images) / row["image"]),
                    "label": str(Path(path_to_masks) / row["mask"])
                }
                for _, row in tr_df.iterrows()
            ]
        if self.model_name == self.unet_name:
            tr_transforms = self.get_tr_transforms(self.unet_properties["img_size"])
            val_transforms = self.get_val_transforms(self.unet_properties["img_size"])
        elif self.model_name == self.unetr_name:
            tr_transforms = self.get_tr_transforms(self.unetr_properties["img_size"])
            val_transforms = self.get_val_transforms(self.unetr_properties["img_size"])
        elif self.model_name == self.swinunetr_name:
            tr_transforms = self.get_tr_transforms(self.swinunetr_properties["img_size"])
            val_transforms = self.get_val_transforms(self.swinunetr_properties["img_size"])
        else:
            raise ValueError(f"invalid 'model_name'. Accepted values: '{self.unet_name}', '{self.unetr_name}, '{self.swinunetr_name}'. ")
        tr_dataset = Dataset(
            tr_data_dicts,
            transform=tr_transforms
        )
        tr_dataloader = DataLoader(
            tr_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True
        )
        if val_samples_csv:
            val_df = pd.read_csv(val_samples_csv)
            val_data_dicts = [
                {
                    "image": str(Path(path_to_images) / row["image"]),
                    "label": str(Path(path_to_masks) / row["mask"])
                }
                for _, row in val_df.iterrows()
            ]
            val_dataset = Dataset(
                val_data_dicts,
                transform=val_transforms
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=1,
                num_workers=num_workers
            )
        # Create model, loss and optimizer
        if device == 'cpu':
            device = torch.device('cpu')
        elif device == 'gpu':
            device = torch.device('cuda:0')
        else:
            raise ValueError("'device' parameter only accepts 'cpu' or 'gpu' value.")
        if self.model_name == self.unet_name:
            model = self.unet
        elif self.model_name == self.unetr_name:
            model = self.unetr
        elif self.model_name == self.swinunetr_name:
            model = self.swinunetr
        else:
            raise ValueError("Invalid value for 'model_name' parameter.")
        model.to(device)
        loss_function = DiceLoss(
            to_onehot_y=True,
            softmax=True
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            1e-4
        )
        dice_metric = DiceMetric(
            include_background=False,
            reduction="none"
        )
        # Run training process
        epoch_loss_values = []
        val_metric_values = []
        best_metric = -1
        best_metric_epoch = -1
        for epoch_idx in tqdm(range(max_epochs)):
            logging.info('-' * 30)
            logging.info(f"epoch: {epoch_idx+1}/{max_epochs}")
            epoch_loss = 0
            model.train()
            start_time = time.time()
            for batch_idx, batch in enumerate(tr_dataloader):
                inputs, labels = (
                    batch["image"].to(device),
                    batch["label"].to(device)
                )
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(
                    outputs,
                    labels
                )
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                if self.verbose:
                    logging.info(f"{batch_idx}/{len(tr_dataset) // tr_dataloader.batch_size}, " f"train_loss: {loss.item():.4f}")
            end_time = time.time()
            epoch_loss /= (batch_idx + 1)
            epoch_loss_values.append(
                {
                    "epoch": epoch_idx + 1,
                    "mean_loss": epoch_loss,
                    "elapsed_time_seconds": end_time - start_time
                }
            )
            logging.info(f"epoch {epoch_idx + 1} average loss: {epoch_loss:.4f}")
            # Run validation
            if (epoch_idx + 1) % val_interval == 0:
                logging.info('-' * 30)
                logging.info(f'running validation at epoch {epoch_idx + 1}')
                model.eval()
                image_paths = []
                mask_paths = []
                inference_time_seconds_per_patch = []
                with torch.no_grad():
                    for batch in val_dataloader:
                        start_time = time.time()
                        inputs, labels = (
                            batch["image"].to(device),
                            batch["label"].to(device)
                        )
                        outputs = model(inputs)
                        outputs = [
                            self.post_prediction(item)
                            for item in decollate_batch(outputs)
                        ]
                        labels = [
                            self.post_label(item)
                            for item in decollate_batch(labels)
                        ]
                        dice_metric(
                            y_pred=outputs,
                            y=labels
                        )
                        image_paths += [
                            item["image_meta_dict"]['filename_or_obj']
                            for item in decollate_batch(batch)
                        ]
                        mask_paths += [
                            item["label_meta_dict"]['filename_or_obj']
                            for item in decollate_batch(batch)
                        ]
                        end_time = time.time()
                        inference_time_seconds_per_patch.append((end_time - start_time) / len(outputs))
                    epoch_metrics = dice_metric.aggregate().squeeze().tolist()
                    dice_metric.reset()
                    mean_metric = sum(epoch_metrics) / len(epoch_metrics)
                    val_metric_values += [
                        {
                            "epoch": epoch_idx + 1,
                            "image": Path(item[0]).name,
                            "mask": Path(item[1]).name,
                            "dice": item[2],
                            "seconds_per_patch": sum(inference_time_seconds_per_patch) / len(inference_time_seconds_per_patch)
                        }
                        for item in zip(image_paths, mask_paths, epoch_metrics)
                    ]
                    # Save model if best until now
                    if mean_metric > best_metric:
                        best_metric = mean_metric
                        best_metric_epoch = epoch_idx + 1
                        torch.save(
                            model.state_dict(),
                            Path(path_to_output) / f"{self.model_name}_best_metric_model.pth"
                        )
                        logging.info("saved new best metric model")
                    logging.info(f"current epoch: {epoch_idx + 1}, current mean dice: {mean_metric:.4f}")
                    logging.info(f"best mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}")
        logging.info(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
        pd.DataFrame(epoch_loss_values).to_csv(
            Path(path_to_output) / "training_losses.csv",
            index=False
        )
        pd.DataFrame(val_metric_values).to_csv(
            Path(path_to_output) / "validation_performance.csv",
            index=False
        )


def predict_sample(path_to_image, path_to_mask, path_to_model,
                   model_name):
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    data_dict = [
        {
            "image": path_to_image,
            "label": path_to_mask
        }
    ]
    trainer = Trainer(model_name)
    if model_name == trainer.unet_name:
        val_transforms = trainer.get_val_transforms(trainer.unet_properties["img_size"])
    elif model_name == trainer.unetr_name:
        val_transforms = trainer.get_val_transforms(trainer.unetr_properties["img_size"])
    elif model_name == trainer.unetr_name:
        val_transforms = trainer.get_val_transforms(trainer.swinunetr_properties["img_size"])
    else:
        raise ValueError("Invalid value for 'model_name' parameter.")
    dataset = Dataset(
        data_dict,
        val_transforms
    )
    dataloader = DataLoader(
        dataset,
        num_workers=4,
        batch_size=1
    )
    if model_name == trainer.unet_name:
        model = trainer.unet
    elif model_name == trainer.unetr_name:
        model = trainer.unetr
    elif model_name == trainer.swinunetr_name:
        model = trainer.swinunetr
    else:
        raise ValueError("Invalid value for 'model_name' parameter.")
    model.load_state_dict(torch.load(path_to_model, weights_only=True))
    model.to(device)
    model.eval()
    dice_metric = DiceMetric(
        include_background=False,
        reduction="mean"
    )
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = (
                batch["image"].to(device),
                batch["label"].to(device)
            )
            outputs = model(inputs)
            outputs = [
                trainer.post_prediction(item)
                for item in decollate_batch(outputs)
            ]
            labels = [
                trainer.post_label(item)
                for item in decollate_batch(labels)
            ]
            dice_metric(
                y_pred=outputs,
                y=labels
            )
        metric = dice_metric.aggregate().item()
        dice_metric.reset()
    print(f"input shape: {inputs.shape}")
    print(f"output shape: {outputs[0].shape}")
    print(f"label shape: {labels[0].shape}")
    print(f"dice: {metric}")
    matshow3d(
        inputs.squeeze().permute(2, 0 ,1),
        title="input volume"
    )
    matshow3d(
        labels[0][1].permute(2, 0 ,1),
        title="ground truth volume"
    )
    matshow3d(
        outputs[0][1].permute(2, 0 ,1),
        title="output volume"
    )
    plt.show()


def make_output_dir(path_to_output, model_name):
    final_dir = f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_{model_name}"
    final_path = Path(path_to_output) / final_dir
    final_path.mkdir()
    return final_path


def serialize_value(value):
    if isinstance(value, Path):
        return str(value)
    else:
        return value


def main():
    parser = argparse.ArgumentParser(
        description="""Script to train a model on specified image-mask
        3d patches from CTC drosophila dataset for binary semantic
        segmentation.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'path_to_images',
        type=str,
        help="""Path to the directory containing the images
        in .tif format."""
    )
    parser.add_argument(
        'path_to_masks',
        type=str,
        help="""Path to the directory containing the masks
        in .tif format."""
    )
    parser.add_argument(
        '--path_to_output',
        type=str,
        default=Path.cwd(),
        help="Path to the output directory."
    )
    parser.add_argument(
        '--path_to_train_patches',
        type=str,
        default=None,
        help="""Path to a csv file containing in each row
        the filenames of image-mask pairs for training.
        The filenames are specified in columns named
        'image' and 'mask', respectively."""
    )
    parser.add_argument(
        '--path_to_val_patches',
        type=str,
        default=None,
        help="""Path to a csv file containing in each row
        the filenames of image-mask pairs for validation.
        The filenames are specified in columns named
        'image' and 'mask', respectively. Both 'path_to_train_patches'
        and 'path_to_val_patches' or none of them have to be specified."""
    )
    parser.add_argument(
        '--model_name',
        type=str,
        choices=['unet', 'unetr', 'swinunetr'],
        default='unet',
        help="Model to be trained."
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help="Number of epochs for training."
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help="Batch size."
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'gpu'],
        default='gpu',
        help="Device for training."
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help="Number of workers for dataloader."
    )
    parser.add_argument(
        '--val_interval',
        type=int,
        default=5,
        help="""Interval of epochs to measure performance on
        the validation set and save best model."""
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help="Add this flag to get verbose log messages."
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help="Seed for reproducibility."
    )
    args = parser.parse_args()
    set_determinism(seed=args.seed)
    path_to_output = make_output_dir(
        args.path_to_output,
        args.model_name
    )
    current_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    logging.basicConfig(
        filename=str(Path(path_to_output) / f"{current_time}_{Path(__file__).stem}.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    trainer = Trainer(args.model_name)
    trainer.verbose = args.verbose
    trainer.train(
        args.path_to_images,
        args.path_to_masks,
        path_to_output,
        training_samples_csv=args.path_to_train_patches,
        val_samples_csv=args.path_to_val_patches,
        max_epochs=args.epochs,
        val_interval=args.val_interval,
        device=args.device,
        num_workers=args.num_workers,
        batch_size=args.batch_size
    )
    with open(Path(path_to_output) / "parameters.json", 'w') as file:
        json_dict = {
            key: serialize_value(value)
            for key, value in vars(args).items()
        }
        json.dump(
            json_dict,
            file,
            indent=4
        )


if __name__ == "__main__":
    main()
