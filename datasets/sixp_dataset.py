"""SixP dataset."""

import glob
import os
from typing import Callable, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import random
import rasterio
import torch
import torchvision.transforms.functional as transF
import torchvision.transforms as torchtrans 
from matplotlib import colors
from rasterio.enums import Resampling
from torch import Tensor
from PIL import Image

from torchgeo.datasets import VisionDataset
# from .utils import check_integrity, extract_archive

from ipdb import set_trace as st

class sixP(VisionDataset):
    """
    Dataset features:

    * RGB aerial images at 0.5 m per pixel spatial resolution (~2,000x2,0000 px)
    * Masks at 0.5 m per pixel spatial resolution (~2,000x2,0000 px)

    Dataset format:

    * images are three-channel geotiffs
    * masks are single-channel geotiffs with the pixel values represent the class

    """  # noqa: E501


    image_root = "images"
    target_root = "labels"

    img_shape = [1000, 1000]

    def __init__(
        self,
        root: str = "data",
        transforms: bool = True,
        normalize: bool = False,
        checksum: bool = False,
        window_size = [224,224],
        fixed_iter = 0,
        unlabeled = False
    ) -> None:
        """Initialize a new tmfrance dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train" or "test"
            transforms: bool, applies random vertical and horizontal flips and rotations to image and target
            normalize: bool, if true normalized image
            checksum: if True, check the MD5 of the downloaded files (may be slow)
        Raises:
            AssertionError: if ``split`` is invalid
        """
        # assert split in self.metadata
        self.root = root
        self.transform = transforms
        self.normalize = normalize
        self.checksum = checksum
        self.unlabeled = unlabeled

        self.fixed_iter = fixed_iter # number of samples on the dataloader
        self.window_size = window_size
        # self.class2idx = {c: i for i, c in enumerate(self.classes)}
        self.files = self._load_files()

        self._verify()


    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        files = self.files[index]
        image = self._load_image(files["image"])
        sample = {"image": image}

        if  not self.unlabeled:
            mask = self._load_target(files["target"])
            sample["mask"] = mask


        if self.transform:
            sample = self._transform(sample)

        # return sample
        if not self.unlabeled:
            return sample['image'], sample['mask']
        
        else:
            return sample['image']


    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)


    def _load_files(self) -> List[Dict[str, str]]:
        """Return the paths of the files in the dataset.

        Returns:
            list of dicts containing paths for each pair of image/mask
        """
        files = []
        images = []

        # for city in self.metadata[self.split]['cities']:
        #     images += glob.glob(os.path.join(self.root, self.image_root, city, '*.tif'))
        images = glob.glob(os.path.join(self.root, self.image_root, '*.png'))

        for image in sorted(images):
            if not self.unlabeled:
                target = image.replace(self.image_root, self.target_root)
                files.append(dict(image=image, target=target))
            else:
                files.append(dict(image=image))

        return files

    def _load_image(self, path: str, shape: Optional[Sequence[int]] = None, resample=False, pos=None) -> Tensor:
        """Load a single image.

        Args:
            path: path to the image
            shape: the (h, w) to resample the image to

        Returns:
            the image
        """
        with rasterio.open(path) as f:
                # Loads entire image
            array = f.read().astype('float32', copy=False)           
            tensor = torch.from_numpy(array)
        
            return tensor / 255.

    def _load_target(self, path: str, resample=False, pos=None) -> Tensor:
        """Load the target mask for a single image.

        Args:
            path: path to the image

        Returns:
            the target mask
        """
        with rasterio.open(path) as f:
            # Loads entire image
            array = f.read().astype('int64', copy=False)
        
            tensor = torch.from_numpy(array)
            # tensor = tensor.to(torch.long)
            return tensor.squeeze(0)

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if checksum fails or the dataset is not downloaded
        """
        for f in self.files:
            if not self.unlabeled == "train":
                if not os.path.isfile(f['image']) or not os.path.isfile(f['target']):
                    raise KeyError('{} is not a file !'.format(f))
            else:
                if not os.path.isfile(f['image']):
                    raise KeyError('{} is not a file !'.format(f))

        return
        
    
    def _get_random_pos(self, window_shape):
        """ Extract of 2D random patch of shape window_shape in the image """
        w, h = window_shape
        W, H = self.img_shape
        x1 = random.randint(0, W - w - 1)
        x2 = x1 + w
        y1 = random.randint(0, H - h - 1)
        y2 = y1 + h
        return x1, x2, y1, y2

    def _transform(self, sample):
        angle = random.choice([0, 90, 180, 270])
        if random.random() > .5:
            for k in sample.keys():
                sample[k] = transF.hflip(sample[k])
        
        if random.random() > .5:
            for k in sample.keys():
                sample[k] = transF.vflip(sample[k])
        for k in sample.keys():
            if sample[k].dim() < 4:
                s = transF.rotate(sample[k].unsqueeze(0), angle)
                sample[k] = s
            else:
                sample[k] = transF.rotate(sample[k], angle)
            
            sample[k] = sample[k].squeeze(0)

        if self.normalize:
            from torchvision.transforms import Normalize
            sample["image"] = Normalize((.5, .5, .5), (.5, .5, .5))(sample["image"])

        return sample



    def plot(
        self,
        sample: Dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        ncols = 2
        image = sample["image"][:3]
        image = image.to(torch.uint8)
        image = image.permute(1, 2, 0).numpy()


        showing_mask = "mask" in sample
        showing_prediction = "prediction" in sample

        cmap = colors.ListedColormap(self.colormap)

        if showing_mask:
            mask = sample["mask"].numpy()
            ncols += 1
        if showing_prediction:
            pred = sample["prediction"].numpy()
            ncols += 1

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(10, ncols * 10))

        axs[0].imshow(image)
        axs[0].axis("off")
        axs[1].axis("off")
        if showing_mask:
            axs[2].imshow(mask, cmap=cmap, interpolation=None)
            axs[2].axis("off")
            if showing_prediction:
                axs[3].imshow(pred, cmap=cmap, interpolation=None)
                axs[3].axis("off")
        elif showing_prediction:
            axs[2].imshow(pred, cmap=cmap, interpolation=None)
            axs[2].axis("off")

        if show_titles:
            axs[0].set_title("Image")

            if showing_mask:
                axs[2].set_title("Ground Truth")
                if showing_prediction:
                    axs[3].set_title("Predictions")
            elif showing_prediction:
                axs[2].set_title("Predictions")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig