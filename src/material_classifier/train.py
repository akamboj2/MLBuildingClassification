from __future__ import print_function, division
from torchvision.models.resnet import resnet50
from torch import nn, optim
from .dataset import *
import torch
import os
from torch.utils.data import DataLoader
from skimage import io, transform
from torchvision import transforms
import numpy as np
import csv


def get_label(image_path, param_path, num_classes, size):
    """
    Gets the label of an image.

    :param image_path: path to the image
    :param param_path: path to the parameter file
    :param num_classes: number of classes
    :param size: size of image
    :return: label of image
    """
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    net = make_frcnn(pretrained=False, num_classes=num_classes)
    if os.path.isfile(param_path):
        net.load_state_dict(torch.load(param_path))

    image = io.imread(image_path)
    image = transform.resize(image, (size, size))
    image = np.expand_dims(np.array(image).transpose((2, 0, 1)), axis=0)
    image = torch.from_numpy(image).float().to(device)
    outputs = net(image)
    _, predicted = torch.max(outputs.data, 1)
    return predicted


def split(root_dir,
          csv_file='./material_classifier/data.csv',
          training='./material_classifier/training.csv',
          validation='./material_classifier/validation.csv'):
    """
    Splits the csv to labeling and training.

    :param root_dir: directory of the labels
    :param csv_file: csv file
    :param training: training csv
    :param validation: validation csv
    """
    results = []
    with open(csv_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            results.append(row)

    val_data = []
    train_data = []

    for row in results:
        if not os.path.isfile(root_dir + row['image']):
            continue
        r = np.random.uniform(0, 1, 1)
        if r <= .1:
            val_data.append(row)
        else:
            train_data.append(row)

    fieldnames = ['image', 'label']

    if len(train_data) + len(val_data) == 0:
        return

    with open(training, 'a', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter=',')
        writer.writerows(train_data)
    with open(validation, 'a', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter=',')
        writer.writerows(val_data)
    with open(csv_file, "w+") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',')
        writer.writeheader()

    del results


def train_resnet(csv_file,
                 root_path,
                 num_classes,
                 param_path,
                 image_size=256,
                 crop_size=256,
                 batch_size=1,
                 epochs=1,
                 lr=10**-3):
    """
    Trains the classifier.

    :param csv_file: path to the csv
    :param root_path: directory of the images
    :param num_classes: number of classes
    :param param_path: path to the parameter file
    :param image_size: size of the input image
    :param crop_size: crop transformation
    :param batch_size: batch size
    :param epochs: number of epochs
    :param lr: learning rate
    """
    torch.manual_seed(0)
    np.random.seed(0)

    split(root_path, csv_file)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    net = resnet50(pretrained=False, num_classes=num_classes)
    net.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    net.train()

    training_dataset = ClassifierDataset(csv_file='./material_classifier/training.csv',
                                         root_dir=root_path,
                                         transformation=transforms.Compose([Rescale((image_size, image_size)),
                                                                            RandomCrop((crop_size, crop_size)),
                                                                            Rescale((image_size, image_size)),
                                                                            Noise('gauss'),
                                                                            Noise('s&p'),
                                                                            ToTensor()]))
    validation_dataset = ClassifierDataset(csv_file='./material_classifier/validation.csv',
                                           root_dir=root_path,
                                           transformation=transforms.Compose([Rescale((image_size, image_size)),
                                                                              ToTensor()]))

    validation_loader = DataLoader(validation_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=0,
                                   drop_last=True
                                   )
    training_loader = DataLoader(training_dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=0,
                                 drop_last=True
                                 )

    state = {
        'nbatches': 0,
        'best': 0,
        'model': net.state_dict()
    }

    if os.path.isfile(param_path):
        state = torch.load(param_path)
        net.load_state_dict(state['model'])

    for epoch in range(epochs):
        running_loss = 00
        if len(training_dataset) == 0:
            print('Empty training dataset.')
        else:
            for _, sample in enumerate(training_loader):
                inputs = sample['image']
                target = sample['label'].long()
                out = net(inputs)
                optimizer.zero_grad()
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.to('cpu').item())

            state['nbatches'] += len(training_loader)

        if len(validation_dataset) == 0:
            print('Empty validation dataset.')
        else:
            val_loss = 0
            total = 0
            correct = 0
            net.requires_grad_(False)
            for _, sample in enumerate(validation_loader):
                with torch.no_grad():
                    inputs = sample['image']
                    target = sample['label'].long()
                    out = net(inputs)
                    loss = criterion(out, target)
                    val_loss += float(loss.to('cpu').item())

                    _, predicted = torch.max(out.data, 1)
                    total += target.flatten().size(0)
                    correct += (predicted == target).sum().item()
            net.requires_grad_(True)

            acc = 100 * correct / total
            val = val_loss/len(validation_dataset)
            progress = {
                'nbatches': state['nbatches'],
                'train_loss': running_loss/len(training_dataset),
                'val_loss': val,
                'accuracy': acc
            }

            print(progress)

            with open('./material_classifier/progress.csv', 'a', newline='') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=progress.keys(), delimiter=',')
                writer.writerow(progress)

            torch.save(state, param_path)
            if acc > state['best']:
                state['best'] = acc
                state['model'] = net.state_dict()
                torch.save(state, "./material_classifier/best_model.txt")

    print('Finished training image classifier')

