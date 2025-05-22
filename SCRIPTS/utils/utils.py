import os
from collections.abc import Sequence

import numpy as np
import pandas as pd
import rasterio
import torch
from ignite.metrics import ConfusionMatrix, Loss
from ignite.metrics.confusion_matrix import cmPrecision, cmRecall
from ignite.metrics.metrics_lambda import MetricsLambda
from onnx2torch import convert
from torch import nn
from torchvision.transforms import v2 as T
from utils.quant_utils.quantize import quant_model

def get_transform(
    mean: Sequence[float],
    std: Sequence[float],
) -> tuple[nn.Sequential, nn.Sequential]:
    transform_train = nn.Sequential(
        T.RandomRotation(90, interpolation=T.InterpolationMode.BILINEAR, fill=0),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.Normalize(
            mean=mean,
            std=std,
            inplace=True,
        ),
    )
    transform_eval = nn.Sequential(
        T.Normalize(
            mean=mean,
            std=std,
            inplace=True,
        ),
    )
    return transform_train, transform_eval


def cmF1(
    average: bool = True,
    precision: MetricsLambda = None,
    recall: MetricsLambda = None,
) -> MetricsLambda:

    f1 = 2.0 * precision * recall / (precision + recall + 1e-15)

    if average:
        f1 = f1.mean().item()

    return f1


def get_metrics(class_count: int, criterion):

    cm_out_fn = lambda output: (
        output[0],
        output[1].argmax(1),
    )

    cm_metric = ConfusionMatrix(
        num_classes=class_count,
        output_transform=lambda output: cm_out_fn(output),
    )
    cm_prec = cmPrecision(cm_metric, average=False)
    cm_rec = cmRecall(cm_metric, average=False)
    cm_F1 = cmF1(average=False, precision=cm_prec, recall=cm_rec)

    metrics = {
        "Precision": cm_prec,
        "Recall": cm_rec,
        "F1": cm_F1,
        "Loss": Loss(criterion),
    }
    return metrics


def read_tiff(filepath: str) -> np.ndarray:
    with rasterio.open(filepath, mode="r", crs=None, transform=None) as i_raster:
        return np.array(i_raster.read()).astype(np.float32)


class ALCDDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_path: str,
        root_dir: str,
    ) -> None:

        self.csv = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.nb_class = 2

    def __len__(self) -> int:
        return len(self.csv)

    def __getitem__(self, idx: int):
        row = self.csv.iloc[idx]

        img_path = os.path.join(self.root_dir, row["in"])
        img = read_tiff(img_path)

        mask_path = os.path.join(self.root_dir, row["out"])
        mask = np.squeeze(read_tiff(mask_path))

        target = np.zeros((self.nb_class, mask.shape[0], mask.shape[1]), dtype=np.float32)
        bin_mask_c = np.where(mask == 0, 1, 0)
        target[0, :, :] += bin_mask_c
        bin_mask_c = np.where(mask == 205, 1, 0)
        target[1, :, :] += bin_mask_c
        # target[0, np.where(mask == 0)] += 1  # background
        # target[1, np.where(mask == 205)] += 1  # clouds


        return img, target


def get_train_test_datasets(data_path, csv_paths):
    train_ds = ALCDDataset(csv_path=csv_paths["train"], root_dir=data_path)
    test_ds = ALCDDataset(csv_path=csv_paths["test"], root_dir=data_path)
    return train_ds, test_ds


def load_onnx_model(onnx_model_path):
    torch_model = convert(onnx_model_path)
    return torch_model


def get_model(name, quant_config=None):
    if name.lower() == "ags_tiny_unet_100k":
        model = load_onnx_model(
            "/nas/PROJETS/HE_EDGE_SPAICE/EXPORT_ARCH_CERN/ags_tiny_unet_100k.onnx"
        )
        # model = load_onnx_model("./MODEL/ags_tiny_unet_100k.onnx")
    elif name.lower() == "ags_tiny_unet_50k":
        model = load_onnx_model(
            "/nas/PROJETS/HE_EDGE_SPAICE/EXPORT_ARCH_CERN/ags_tiny_unet_50k.onnx"
        )
        # model = load_onnx_model("./MODEL/ags_tiny_unet_50k.onnx")
    else:
        raise ValueError(
            f"{name=}, should be `ags_tiny_unet_100k` or `ags_tiny_unet_50k`"
        )

    if quant_config is not None:
        model = quant_model(quant_config, model)

    return model
