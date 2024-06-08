
from __future__ import print_function, division

import os
from pathlib import Path
import csv
import numpy as np
import pandas as pd
import torch
from skimage import io, transform
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    """
    Dataset of image segmentation.

    Each sample is a dictionary containing:
        'image': path to the original image
        'target': pixel-wise label
    """

    def __init__(self, csv_file, root_dir, transformation=None):
        """
        Constructor

        :param csv_file (string): Path to the csv file with annotations.
        :param root_dir (string): Directory with all the images.
        :param transformation (callable, optional): Optional transform to be applied  on a sample.
        """
        self.bubbles = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transformation

    def __len__(self):
        """
        Gets the size

        :return: Length of the dataset
        """
        return len(self.bubbles)

    def __getitem__(self, idx):
        """
        Gets a label

        :param idx: index of the sample
        :return: sample
        """
        img_name = self.bubbles.iloc[idx, 0]
        image = io.imread(self.root_dir + img_name + '.jpg')
        target = np.load(self.root_dir + img_name + '.npy')

        sample = {'image': image, 'target': target}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Noise(object):
    """
    Applies noise to the image.

    This piece of code was extracted from the following source with few modifications:
    https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
    """
    def __init__(self, noise):
        """

        :param noise: One of the following strings, selecting the type of noise to add:
        'gauss'     Gaussian-distributed additive noise.
        'poisson'   Poisson-distributed noise generated from the data.
        's&p'       Replaces random pixels with black or white.
        'speckle'   Multiplicative noise using out = image + n*image,where
                    n is uniform noise with specified mean & variance.
        """
        self.noise = noise

    def __call__(self, sample):
        """
        Applies the noise tot he sample

        :param sample: dictionary containing 'image' and 'target'
        :return: transformed sample
        """
        image, target = sample['image'], sample['target']
        if self.noise == "gauss":
            row, col, ch = image.shape
            mean = 0
            var = 4
            sigma = var**0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            noisy = image + gauss
            return {'image': noisy, 'target': target}
        elif self.noise == "s&p":
            row, col, ch = image.shape
            s_vs_p = 0.5
            amount = 0.004
            out = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = np.array([np.random.randint(0, i - 1, int(num_salt))
                               for i in image.shape])
            out[coords] = 255

            # Pepper mode
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = np.array([np.random.randint(0, i - 1, int(num_pepper))
                               for i in image.shape])
            out[coords] = 0
            return {'image': out, 'target': target}
        elif self.noise == "poisson":
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(image * vals) / float(vals)
            return {'image': noisy, 'target': target}
        elif self.noise =="speckle":
            row, col, ch = image.shape
            gauss = np.random.randn(row, col, ch)
            noisy = image + image * gauss
            return {'image': noisy, 'target': target}
        else:
            raise Exception("Invalid noise")


class Rescale(object):
    """
    Rescales the sample
    """

    def __init__(self, output_size):
        """
        Constructor

        :param output_size: Desired output size. If tuple, output is matched to output_size. If int, smaller of image edges is matched to output_size keeping aspect ratio the same.
        """
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        """
        Rescales the image to the desired size.

        :param sample: dictionary containing 'image' and 'target'
        :return: transformed sample
        """
        image, output = sample['image'], sample['target']

        new_h, new_w = self.output_size

        new = output
        step = 25
        while new.shape[0] > new_h or new.shape[1] > new_w:
            x_step = step if new.shape[0] > new_h + step else (1 if new.shape[0] > new_h else 0)
            y_step = step if new.shape[1] > new_w + step else (1 if new.shape[1] > new_w else 0)
            old = new
            new = np.zeros((old.shape[0] - x_step, old.shape[1] - y_step))
            for i in range(new.shape[0]):
                for j in range(new.shape[1]):
                    filter = np.reshape(old[i:i + x_step + 1, j:j + y_step + 1], (-1,))
                    counts = np.bincount(filter.astype(int, copy=False))
                    new[i, j] = np.argmax(counts)

        while new.shape[0] < new_h or new.shape[1] < new_w:
            x_step = True if new.shape[0] < new_h else False
            y_step = True if new.shape[1] < new_w else False
            if x_step:
                idx = int(np.random.uniform(0, new.shape[0]))
                new = np.vstack([new[:idx], np.reshape(new[idx], (1, -1)), new[idx:]])
            if y_step:
                idx = int(np.random.uniform(0, new.shape[0]))
                new = np.hstack([new[:, :idx], np.reshape(new[:, idx], (-1, 1)), new[:, idx:]])
        output = new

        image = transform.resize(image, self.output_size)
        return {'image': image, 'target': output}


class RandomCrop(object):
    """
    Transformation of a sample. Crop randomly the image in a sample.
    """

    def __init__(self, output_size):
        """
        Constructor

        :param output_size: Desired output size (tuple)
        """
        self.output_size = output_size

    def __call__(self, sample):
        """
        Returns the sample cropped.

        :param sample: dictionary containing 'image' and 'target'
        :return: cropped sample
        """
        image, target = sample['image'], sample['target']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        target = target[top: top + new_h,
                        left: left + new_w]

        return {'image': image, 'target': target}


class ToTensor(object):
    """
    This class convert ndarrays in sample to Tensors.
    """

    def __call__(self, sample):
        """
        Convert ndarrays in sample to Tensors.

        :param sample: dictionary containing 'image' and 'target'
        :return: sample in tensors
        """
        image, target = sample['image'], sample['target']

        image = np.array(image).transpose((2, 0, 1))
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        return {'image': torch.from_numpy(image).float().to(device),
                'target': torch.from_numpy(target).float().to(device)}

