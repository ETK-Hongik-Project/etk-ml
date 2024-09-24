import os
from typing import Literal

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.io as tio
import torchvision.transforms.v2 as transforms
from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple
from tqdm import tqdm


def loadMetadata(filename, silent=False):
    try:
        if not silent:
            tqdm.write('\tReading metadata from %s ...' % filename)
        metadata = sio.loadmat(filename, squeeze_me=True,
                               struct_as_record=False)
    except:
        tqdm.write('\tFailed to read the meta file "%s"!' % filename)
        return None
    return metadata


class GTKDataset(Dataset):
    def __init__(self, data_path: str,
                 split: Literal['train', 'val', 'test'] = 'train',
                 img_size: tuple = (112, 112),
                 channel: Literal[1, 3] = 1):
        self.data_path = data_path
        self.img_size = img_size
        self.channel = channel

        tqdm.write("Loading Dataset...")

        meta_path = os.path.join(data_path, 'metadata.mat')
        if meta_path is None or not os.path.isfile(meta_path):
            raise RuntimeError(f'NonExist File : {meta_path}')

        self.meta_file = loadMetadata(meta_path)
        if self.meta_file is None:
            raise RuntimeError("Couldn't Read Metafile")

        # Augmenatation & Transfromation(Normalization)
        # TODO:? Random Rotation?
        self.resize = transforms.Resize(self.img_size)
        self.augmentation = transforms.Compose([
            transforms.Lambda(lambda img: img +
                              torch.rand_like(img) * 3 * torch.randn(1)),
            transforms.ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0,
                hue=0,
            ),
            transforms.Lambda(lambda img: torch.clamp(img, 0, 255))
        ])
        self.normalization = transforms.Lambda(lambda img: img / 127.5 - 1)

        self.mask = None
        self.split = split
        if self.split == 'train':
            self.mask = self.meta_file['labelTrain']
        elif self.split == 'val':
            self.mask = self.meta_file['labelVal']
        elif self.split == 'test':
            self.mask = self.meta_file['labelTest']

        self.indices = np.argwhere(self.mask)[:, 0]
        tqdm.write(f'Loaded {split} Dataset: {len(self.indices)} Records.')

    def load_image(self, path: str):
        try:
            if self.channel == 1:
                # Don't Need to Read PIL image and transform to the tensor.
                img = tio.read_image(path, tio.ImageReadMode.GRAY)
            elif self.channel == 3:
                img = tio.read_image(path)
        except:
            raise RuntimeError(f"Couldn't Read IMG : f{path}")
        return img.float()

    def gaze_to_direction(self, gaze: np.ndarray, c: int = 6) -> torch.Tensor:
        """
        X, Y \in [-25, 25]
        Center(4) : |X| < C, |Y| < C
        Left(2) : X < -C, |X| > |Y|
        Right(3) : X > C, |X| > |Y|
        Top(0) : Y > C, |X| < |Y|
        Bottom(1) : Y < -C, |X| < |Y|
        """
        X, Y = gaze[0], gaze[1]

        direction = None
        if np.abs(X) <= c and np.abs(Y) <= c:
            direction = 4  # Center
        elif X < -c and np.abs(X) > np.abs(Y):
            direction = 2  # Left
        elif X > c and np.abs(X) > np.abs(Y):
            direction = 3  # Right
        elif Y > c and np.abs(X) < np.abs(Y):
            direction = 0  # Top
        elif Y < -c and np.abs(X) < np.abs(Y):
            direction = 1  # Bottom

        return torch.tensor(direction)

    def create_grid(self, g: np.ndarray, sx: int, sy: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create 2 Channel Info for Face Position.
           One of channel Contain Row info, the Other contains Column Info

        Args:
            g (array): [x, y, w, h]
            sx (int): size of width for new img
            sy (int): size of height for new img
        """
        x_start = g[0] * 2 / 25 - 1
        x_end = (g[0] + g[2]) * 2 / 25 - 1
        y_start = g[1] * 2 / 25 - 1
        y_end = (g[1] + g[3]) * 2 / 25 - 1
        linx = np.linspace(x_start, x_end, sx, dtype=np.float32)
        liny = np.linspace(y_start, y_end, sy, dtype=np.float32)
        return np.meshgrid(linx, liny)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]

        face_img_path = os.path.join(self.data_path,
                                     f"{self.meta_file['labelRecNum'][idx]:05d}/appleFace/{self.meta_file['frameIndex'][idx]:05d}.jpg")

        # Load and Transform Gray Img
        face_img = self.load_image(face_img_path)
        face_img = self.resize(face_img)
        if self.split == "train":
            face_img = self.augmentation(face_img)
        face_img = self.normalization(face_img)  # (1, 112, 112)

        # Create Grid Info
        grid = self.meta_file['labelFaceGrid'][idx, :]
        grid_x, grid_y = self.create_grid(grid, *self.img_size)
        grid_x = torch.from_numpy(np.expand_dims(grid_x, 0))  # (1, 112, 112)
        grid_y = torch.from_numpy(np.expand_dims(grid_y, 0))  # (1, 112, 112)

        img = torch.concat([face_img, grid_x, grid_y], dim=0)  # (3, 112, 112)

        gaze = np.array([
            self.meta_file['labelDotXCam'][idx], self.meta_file['labelDotYCam'][idx]
        ], np.float32)
        direction = self.gaze_to_direction(gaze)

        return img, direction

if __name__ == "__main__":
    data_path = "./Processed"
    trainset = GTKDataset(data_path=data_path, split='train')
    valset = GTKDataset(data_path=data_path, split='val')
    testset = GTKDataset(data_path=data_path, split='test')