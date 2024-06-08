from __future__ import print_function, division
import pandas as pd
import torch
from skimage import  transform
from torch.utils.data import Dataset
from PIL import Image
import re
import ast
import numpy as np


def str2array(s):
    """
    Transforms the string to an array
    :param s: string
    :return: array
    """
    s = re.sub('\[ +', '[', s.strip())
    s = re.sub('[,\s]+', ', ', s)
    return np.array(ast.literal_eval(s))


class ObjectDataset(Dataset):
    """
    Object detection dataset

    Each sample is a dictionary containing:
        'image': path to the original image
        'labels': array of labels and boxes [label, x0, y0, x1, y1]
    """

    def __init__(self, csv_file, root_dir, transformation=None):
        """
        :param csv_file: csv file
        :param root_dir: directory of the samples
        :param transformation: transformations
        """
        self.samples = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transformation

    def __len__(self):
        """
        :return: length of the sataset
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns a sample
        :param idx: index of the sample
        :return: sample
        """
        img_path = self.root_dir + self.samples.iloc[idx, 0]
        cell = self.samples.iloc[idx, 1]
        tags = np.array(str2array(cell))

        if len(tags) >= 1:
            labels = tags[:, 0]
            boxes = tags[:, 1:]
        else:
            labels = []
            boxes = []

        image = np.array(Image.open(img_path).convert('RGB'))

        sample = {'image': image, 'labels': labels, 'boxes': boxes}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    """Transformation of the size of a sample"""

    def __init__(self, output_size):
        """
        :param output_size: Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
        """
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        """
        Rescales the image to the desired size.
        :param sample: sample
        :return: transformed sample
        """
        image, boxes = sample['image'], sample['boxes']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size, self.output_size
        # Box is [x1, y1, x2, y2]
        if len(boxes) > 0:
            boxes[:, 0] = boxes[:, 0]*new_w / w
            boxes[:, 1] = boxes[:, 1]*new_h / h
            boxes[:, 2] = boxes[:, 2]*new_w / w
            boxes[:, 3] = boxes[:, 3]*new_h / h

        image = transform.resize(image, (self.output_size, self.output_size))
        return {'image': image, 'boxes': boxes, 'labels': sample['labels']}


class RandomCrop(object):
    """Transformation of a sample. Crop randomly the image in a sample."""

    def __init__(self, output_size):
        """
        :param output_size: Desired output size (tuple)
        """
        self.output_size = output_size

    def __call__(self, sample):
        """
        Returns the sample cropped.
        :param sample: sample
        :return: cropped sample
        """
        image, boxes, labels = sample['image'], sample['boxes'], sample['labels']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size, self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        # Assuming that x and y are 0 at the upper left corner
        if len(boxes) > 0:
            boxes[:, [0, 2]] += top
            boxes[:, [1, 3]] += left

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'boxes': boxes, 'labels': labels}


class ToTensor(object):
    """This class convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        """
        Convert ndarrays in sample to Tensors.
        :param sample: sample
        :return: sample in tensors
        """
        image, box, label = sample['image'], sample['boxes'], sample['labels']

        image = np.array(image).transpose((2, 0, 1))

        return {'image': torch.from_numpy(image).float(),
                'labels': torch.ones(0).type(torch.int64) if len(label) == 0 else torch.from_numpy(label).type(torch.int64),
                'boxes':  torch.zeros(0, 4) if len(label) == 0 else torch.from_numpy(box).float()}
