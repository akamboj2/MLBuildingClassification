from __future__ import print_function, division

import pandas as pd
import torch
from skimage import io, transform
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class ClassifierDataset(Dataset):
    """
     Classifier dataset.

    Each sample is a dictionary containing:
    'image': path to the original image
    'label': label
    """

    def __init__(self, csv_file, root_dir, transformation=None):
        """
        Constructor

        :param csv_file: path to csv file
        :param root_dir: root firectory of the samples
        :param transformation: transformations
        """
        self.samples = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transformation

    def __len__(self):
        """
        Gets the length

        :return: Length of the dataset
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns a sample

        :param idx: index of the sample
        :return: sample
        """
        img_path = self.root_dir + self.samples.iloc[idx, 0]
        label = np.array(int(self.samples.iloc[idx, 1]))

        try:
            image = np.array(Image.open(img_path).convert('RGB'))
        except:
            return self[(idx + 1) % self.__len__()]

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    """
    Transformation of the size of a sample
    """

    def __init__(self, output_size):
        """

        :param output_size: Desired output size. If tuple, output is matched to output_size. If int, smaller of image edges is matched to output_size keeping aspect ratio the same.
        """
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        """
        Rescales the image to the desired size.

        :param sample: sample
        :return: transformed sample
        """
        image = sample['image']
        image = transform.resize(image, self.output_size)
        return {'image': image, 'label': sample['label']}


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

        :param sample: dictionary containing 'image' and 'label'
        :return: transformed sample
        """
        image, target = sample['image'], sample['label']
        if self.noise == "gauss":
            row, col, ch = image.shape
            mean = 0
            var = 0.01
            sigma = var**0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            noisy = image + gauss
            return {'image': noisy, 'label': target}
        elif self.noise == "s&p":
            s_vs_p = 0.5
            amount = 0.004
            out = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in image.shape]
            out[coords] = 1

            # Pepper mode
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in image.shape]
            out[coords] = 0
            return {'image': out, 'label': target}
        elif self.noise == "poisson":
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(image * vals) / float(vals)
            return {'image': noisy, 'label': target}
        elif self.noise =="speckle":
            row, col, ch = image.shape
            gauss = np.random.randn(row, col, ch)
            noisy = image + image * gauss
            return {'image': noisy, 'label': target}
        else:
            raise Exception("Invalid noise")


class RandomCrop(object):
    """Transformation of a sample. Crop randomly the image in a sample."""

    def __init__(self, output_size):
        """
        Constructor

        :param output_size: Desired output size (tuple)
        """
        self.output_size = output_size

    def __call__(self, sample):
        """
        Returns the sample cropped.

        :param sample: sample
        :return: cropped sample
        """
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'label': label}


class ToTensor(object):
    """
    This class convert ndarrays in sample to Tensors.
    """

    def __call__(self, sample):
        """
        Convert ndarrays in sample to Tensors.

        :param sample: sample
        :return: sample in tensors
        """
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        image = np.array(image).transpose((2, 0, 1))

        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        return {'image': torch.from_numpy(image).float().to(device),
                'label': torch.from_numpy(label).type(torch.int64).to(device)}
