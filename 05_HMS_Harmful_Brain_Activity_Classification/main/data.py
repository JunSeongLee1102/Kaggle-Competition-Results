import os
import sys
import gc
from argparse import Namespace
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import lightning.pytorch as pl
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import albumentations as albu
from sklearn.model_selection import KFold, GroupKFold

import pywt, librosa

if os.path.isdir("/kaggle"):
    a = 0
    ROOT = "/kaggle"
else:
    sys.path.append("../main")
    ROOT = ".."
from utils import grid_search, kfold
from preprocess import load_train, load_test


class HMS_DS(Dataset):
    def __init__(
        self,
        hp,
        meta_data,
        targets=None,
        augment=False,
        force_hor_flip=None,
        mode="train",
        specs=None,
        eeg_specs=None,
    ):
        self.hp = hp
        self.meta_data = meta_data
        self.targets = targets
        self.augment = augment
        self.hor_flip = 0.0
        if force_hor_flip is not None:
            self.hor_flip = force_hor_flip
        elif "hor_flip" in hp.augment_species.keys():
            self.hor_flip = hp.augment_species["hor_flip"]["p"]
        self.mode = mode
        self.specs = specs
        self.eeg_specs = eeg_specs
        self.cache_data = hp.cache_data

        self.Xs = []
        self.ys = []
        self.categories = []
        if self.cache_data:
            self.Xs, self.ys, self.categories = self._generate_data(
                np.arange(0, len(self.meta_data))
            )

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, index):
        if self.cache_data:
            return {
                "x": self.Xs[index],
                "y": self.ys[index],
                "category": self.categories[index],
            }
        else:
            return self.__getitems__([index])

    def __getitems__(self, indices):
        if self.cache_data:
            X = self.Xs[indices].copy()
            y = self.ys[indices].copy()
            categories = self.categories[indices]
        else:
            X, y, categories = self._generate_data(indices)
        if self.mode != "test":
            if self.mode == "train":
                if self.augment:
                    X, y = self.__augment(X, y)
            if random.random() <= self.hor_flip:
                X = np.flip(X, axis=2).copy()
            return {"x": X, "y": y, "category": categories}
        else:
            return X

    def _generate_data(self, indexes):
        num_chs = np.shape(self.eeg_specs[self.meta_data.iloc[0].eeg_id])[-1]
        X = np.zeros((len(indexes), 128, 256, num_chs + 4), dtype="float32")
        categories = np.zeros((len(indexes), 1), dtype=int)
        y = np.zeros((len(indexes), 6), dtype="float32")
        img = np.ones((128, 256), dtype="float32")
        for j, i in enumerate(indexes):
            row = self.meta_data.iloc[i]
            categories[j] = row["vote_category"]
            if self.mode == "test":
                r = 0
            else:
                r = int((row["min"] + row["max"]) // 4)
            for k in range(4):
                # EXTRACT 300 ROWS OF SPECTROGRAM
                img = self.specs[row.spec_id][r : r + 300, k * 100 : (k + 1) * 100].T

                # LOG TRANSFORM SPECTROGRAM
                img = np.clip(img, np.exp(-4), np.exp(8))
                img = np.log(img)

                # STANDARDIZE PER IMAGE
                ep = 1e-6
                m = np.nanmean(img.flatten())
                s = np.nanstd(img.flatten())
                img = (img - m) / (s + ep)
                img = np.nan_to_num(img, nan=0.0)

                # CROP TO 256 TIME STEPS
                X[j, 14:-14, :, k] = img[:, 22:-22] / 2.0

            # EEG SPECTROGRAMS
            img = self.eeg_specs[row.eeg_id]
            X[j, :, :, 4:] = img
            # X[j, :, :] = img
            if self.mode != "test":
                y[j,] = row[self.targets]
        x1 = [X[:, :, :, i : i + 1] for i in range(4)]
        x1 = np.concatenate(x1, axis=1)
        x2 = [X[:, :, :, i + 4 : i + 5] for i in range(num_chs)]
        x2 = np.concatenate(x2, axis=1)
        if self.hp.use_kaggle_spectrograms & self.hp.use_eeg_spectrograms:
            X = np.concatenate([x1, x2], axis=1)
        elif self.hp.use_eeg_spectrograms:
            X = x2
        else:
            X = x1
        return X, y, categories

    def _random_transform(self, img):
        augment_keys = self.hp.augment_species.keys()
        augment_list = []
        if "hor_flip" in augment_keys:
            augment_list.append(
                albu.HorizontalFlip(**self.hp.augment_species["hor_flip"])
            )
        if "random_br_cont" in augment_keys:
            augment_list.append(
                albu.RandomBrightnessContrast(
                    **self.hp.augment_species["random_br_cont"]
                )
            )
        if "down_scale" in augment_keys:
            augment_list.append(
                albu.augmentations.transforms.Downscale(
                    **self.hp.augment_species["down_scale"]
                )
            )
        if "emboss" in augment_keys:
            augment_list.append(
                albu.augmentations.transforms.Emboss(
                    **self.hp.augment_species["emboss"]
                )
            )
        if "gauss_noise" in augment_keys:
            augment_list.append(
                albu.augmentations.transforms.GaussNoise(
                    **self.hp.augment_species["gauss_noise"]
                )
            )
        if "invert" in augment_keys:
            augment_list.append(
                albu.augmentations.transforms.InvertImg(
                    **self.hp.augment_species["invert"]
                )
            )
        if "multi_noise" in augment_keys:
            augment_list.append(
                albu.augmentations.transforms.MultiplicativeNoise(
                    **self.hp.augment_species["multi_noise"]
                )
            )
        if "pixel_dropout" in augment_keys:
            augment_list.append(
                albu.augmentations.transforms.PixelDropout(
                    **self.hp.augment_species["pixel_dropout"]
                )
            )
        if "random_br" in augment_keys:
            augment_list.append(
                albu.augmentations.transforms.RandomBrightnessContrast(
                    **self.hp.augment_species["random_br"]
                )
            )

        if "random_gamma" in augment_keys:
            augment_list.append(
                albu.augmentations.transforms.RandomGamma(
                    **self.hp.augment_species["random_gamma"]
                )
            )
        if "random_grid_shuffle" in augment_keys:
            augment_list.append(
                albu.augmentations.transforms.RandomGridShuffle(
                    **self.hp.augment_species["random_grid_shuffle"]
                )
            )
        if "ring_overshoot" in augment_keys:
            augment_list.append(
                albu.augmentations.transforms.RingingOvershoot(
                    **self.hp.augment_species["ring_overshoot"]
                )
            )
        if "sharpen" in augment_keys:
            augment_list.append(
                albu.augmentations.transforms.Sharpen(
                    **self.hp.augment_species["sharpen"]
                )
            )
        if "solarize" in augment_keys:
            augment_list.append(
                albu.augmentations.transforms.Solarize(
                    **self.hp.augment_species["solarize"]
                )
            )
        if "super_pixel" in augment_keys:
            augment_list.append(
                albu.augmentations.transforms.Superpixels(
                    **self.hp.augment_species["super_pixel"]
                )
            )
        if "unsharp" in augment_keys:
            augment_list.append(
                albu.augmentations.transforms.UnsharpMask(
                    **self.hp.augment_species["unsharp"]
                )
            )
        if "xy_mask" in augment_keys:
            augment_list.append(albu.XYMasking(**self.hp.augment_species["xy_mask"]))
        composition = albu.Compose(augment_list)
        return composition(image=img)["image"]

    def _mixup(self, img_batch, y_batch, clip=[0, 1]):
        if random.random() > self.hp.augment_species["p_mixup"]:
            return img_batch, y_batch
        indices = torch.randperm(len(img_batch))
        shuffled_img_batch = img_batch[indices].copy()
        shuffled_y_batch = y_batch[indices].copy()

        lam = np.random.uniform(clip[0], clip[1])
        img_batch = img_batch * lam + shuffled_img_batch * (1 - lam)
        y_batch = y_batch * lam + shuffled_y_batch * (1 - lam)
        return img_batch, y_batch

    # https://www.sciencedirect.com/science/article/abs/pii/S0893608021002288
    def _cutcat(self, img_batch, y_batch, clip=[0, 1]):
        if random.random() > self.hp.augment_species["p_cutcat"]:
            return img_batch, y_batch
        indices = torch.randperm(len(img_batch))
        shuffled_img_batch = img_batch[indices].copy()
        shuffled_y_batch = y_batch[indices].copy()
        img_x_len = len(img_batch[0, 0])
        bord = np.linspace(0, img_x_len, 5, dtype=int)
        for i in range(1, 4, 2):
            img_batch[:, :, bord[i] : bord[i + 1]] = shuffled_img_batch[
                :, :, bord[i] : bord[i + 1]
            ]
        y_batch = (y_batch + shuffled_y_batch) / 2
        return img_batch, y_batch

    def __augment(self, img_batch, y_batch):
        for i in range(img_batch.shape[0]):
            img_batch[i,] = self._random_transform(img_batch[i,])
        if "p_mixup" in self.hp.augment_species.keys():
            img_batch, y_batch = self._mixup(img_batch, y_batch)
        if "p_cutcat" in self.hp.augment_species.keys():
            img_batch, y_batch = self._cutcat(img_batch, y_batch)
        return img_batch, y_batch


class HMS_DM(pl.LightningDataModule):
    def __init__(
        self,
        hp: Namespace,
        n_workers: int = 0,
        train_meta_csv_path: str | None = None,
        train_spec_path: str | None = None,
        train_eeg_path: str | None = None,
        test_meta_csv_path: str | None = None,
        test_spec_path: str | None = None,
        test_eeg_path: str | None = None,
    ):
        super().__init__()
        args = hp.read_spec_files, hp.read_eeg_spec_files
        self.hp = hp
        if train_meta_csv_path:
            (
                self.train_meta_df,
                self.targets,
                self.train_spectrograms,
                self.train_eegs,
            ) = load_train(train_meta_csv_path, train_spec_path, train_eeg_path, *args)
            fold_idxs = kfold(self.train_meta_df, hp.n_folds)[hp.fold]
            self.train_split_df = self.train_meta_df.iloc[fold_idxs[0]]
            self.valid_split_df = self.train_meta_df.iloc[fold_idxs[1]]
            self.kwargs = {
                "batch_size": hp.batch_size,
                "num_workers": n_workers,
                "pin_memory": bool(n_workers),
            }
        if test_meta_csv_path:
            self.test_meta_df, self.test_spectrograms, self.test_eegs = load_test(
                test_meta_csv_path, test_spec_path, test_eeg_path
            )

    def collate_fn(self, data):
        collated_x = []
        collated_y = []
        collated_category = []
        for i in range(0, len(data["x"])):
            collated_x.append(torch.from_numpy(data["x"][i]))
            collated_y.append(torch.from_numpy(data["y"][i]))
            collated_category.append(torch.from_numpy(data["category"][i]))
        return {
            "x": torch.stack(collated_x),
            "y": torch.stack(collated_y),
            "category": torch.stack(collated_category),
        }

    def train_dataloader(self):
        assert self.train_split_df is not None
        ds = HMS_DS(
            self.hp,
            self.train_split_df,
            self.targets,
            self.hp.augment,
            None,
            "train",
            self.train_spectrograms,
            self.train_eegs,
        )
        return DataLoader(
            ds,
            collate_fn=self.collate_fn,
            **(self.kwargs | {"shuffle": True}),
            persistent_workers=True
        )

    def val_dataloader(self):
        assert self.valid_split_df is not None
        dl = DataLoader(
            HMS_DS(
                self.hp,
                self.valid_split_df,
                self.targets,
                False,
                0.0,
                "valid",
                self.train_spectrograms,
                self.train_eegs,
            ),
            collate_fn=self.collate_fn,
            **(self.kwargs | {"shuffle": False}),
            persistent_workers=True
        )
        dl = (
            [
                dl,
                DataLoader(
                    HMS_DS(
                        self.hp,
                        self.valid_split_df,
                        self.targets,
                        False,
                        1.0,
                        "valid",
                        self.train_spectrograms,
                        self.train_eegs,
                    ),
                    collate_fn=self.collate_fn,
                    **(self.kwargs | {"shuffle": False}),
                    persistent_workers=True
                ),
            ]
            if self.hp.flip_val
            else dl
        )
        return dl

    def predict_dataloader(self):
        assert self.test_meta_df is not None
        ds = HMS_DS(
            self.hp,
            self.test_meta_df,
            self.targets,
            False,
            self.test_spectrograms,
            self.test_eegs,
        )
        return DataLoader(ds, **(self.kwargs) | {"shuffle": False})
