from __future__ import print_function, division
from torchvision.models.detection import fasterrcnn_resnet50_fpn as make_frcnn
from .dataset import *
import torch
import os
from torch.utils.data import DataLoader
from skimage import io, transform
from torchvision import transforms
import csv


def get_objects(img_path, param_path, num_classes=1, size=720):
    """
    Gts the objects found
    :param img_path: path to image
    :param param_path: path to parameter file
    :param num_classes: number of classes
    :param size: size of image as input
    :return:
    """
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    net = make_frcnn(pretrained=False, num_classes=num_classes).to(device)
    net.eval()

    if os.path.isfile('./parameters.txt'):
        net.load_state_dict(torch.load(param_path))

    image = io.imread(img_path)
    image = transform.resize(image, (size, size))
    image = np.expand_dims(np.array(image).transpose((2, 0, 1)), axis=0)
    image = torch.from_numpy(image).float().to(device)

    objects = net(image)
    for key in objects.keys():
        objects[key].to('cpu')
    return objects


def split(csv_file='./object_detection/data.csv',
          training='./object_detection/training.csv',
          validation='./object_detection/validation.csv'):
    """
    Splits the dataset
    :param csv_file: path to csv
    :param training: path to training csv
    :param validation: path to validation csv
    """
    results = []
    with open(csv_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            results.append(row)

    val_data = []
    train_data = []
    extra = []

    for row in results:
        if row['tags'] != "[]":
            r = np.random.uniform(0, 1, 1)
            if r <= .1:
                val_data.append(row)
            else:
                train_data.append(row)
        else:
            extra.append(row)

    fieldnames = ['image', 'tags']

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
    with open('./object_detection/extra.csv', 'a', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter=',')
        writer.writerows(extra)

    del results


def train_f_rcnn(csv_file, num_classes, param_path, image_size=256,
                 crop_size=256, batch_size=1, nepochs=1, lr=0.01,
                 root_dir='./resources/object_detection/labeled/'):
    """
    Trains the Faster RCNN. Training results are printed to the procress.csv

    :param csv_file: path to csv file
    :param num_classes: number of classes
    :param param_path: path to parameter file
    :param image_size: image used as input size
    :param crop_size: crop tranformation size
    :param batch_size: batch size
    :param nepochs: number of epochs
    :param lr: learning rate (ignored)
    :param root_dir: root directory of images
    """
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    net = make_frcnn(pretrained=False, num_classes=num_classes).to(device)
    split(csv_file)

    training_dataset = ObjectDataset(csv_file='./object_detection/training.csv',
                                     root_dir=root_dir,
                                     transformation=transforms.Compose([
                                            Rescale(image_size),
                                            RandomCrop(crop_size),
                                            Rescale(image_size),
                                            ToTensor()
                                     ]))
    validation_dataset = ObjectDataset(csv_file='./object_detection/validation.csv',
                                       root_dir=root_dir,
                                       transformation=transforms.Compose([
                                            Rescale(image_size),
                                            RandomCrop(crop_size),
                                            Rescale(image_size),
                                            ToTensor()
                                       ]))

    def _collate_fn(data):
        indexes = [np.array(data[i]['labels']).size > 0 for i in range(len(data))]
        lst = [data[i].pop('image', None) for i in range(len(data)) if indexes[i]]
        images = torch.stack(lst).to(device)
        for i in range(len(data)):
            for key in data[i].keys():
                if key == 'labels':
                    data[i][key] = torch.zeros(data[i][key].shape, dtype=torch.int64)
                data[i][key] = data[i][key].to(device)
        objects = [data[i] for i in range(len(data)) if indexes[i]]
        new_data = {'image': images, 'objects': objects}
        del data
        return new_data

    validation_loader = DataLoader(validation_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   collate_fn=_collate_fn
                                   # , num_workers=1
                                   )
    training_loader = DataLoader(training_dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 collate_fn=_collate_fn
                                 # , num_workers=1
                                 )

    state = {
        'nbatches': 0,
        'best': {'loss_classifier': 0.0,
                 'loss_box_reg': 0.0,
                 'loss_objectness': 0.0,
                 'loss_rpn_box_reg': 0.0},
        'model': net.state_dict()
    }

    if os.path.isfile(param_path):
        state = torch.load(param_path)
        net.load_state_dict(state['model'])

    for epoch in range(nepochs):

        running_loss = {'loss_classifier': 0.0,
                        'loss_box_reg': 0.0,
                        'loss_objectness': 0.0,
                        'loss_rpn_box_reg': 0.0}

        for _, sample in enumerate(training_loader):

            one = sample['image'].to(device)
            second = sample['objects']
            loss = net(one, second)

            one.to('cpu')
            for i in range(len(second)):
                for key in second[i].keys():
                    second[i][key].to('cpu')

            for key in running_loss.keys():
                running_loss[key] += float(loss[key].to('cpu'))
            del loss

        state['nbatches'] += len(training_loader)

        val_loss = {'loss_classifier': 0.0,
                    'loss_box_reg': 0.0,
                    'loss_objectness': 0.0,
                    'loss_rpn_box_reg': 0.0}

        net.requires_grad_(False)
        for _, sample in enumerate(validation_loader):
            with torch.no_grad():
                input = sample['image'].to(device)
                target = sample['objects']

                loss = net(input, target)
                for key in running_loss.keys():
                    val_loss[key] += float(loss[key].to('cpu'))
        net.requires_grad_(True)

        progress = {
            'nbatches': state['nbatches'],
            'train_loss_classifier': running_loss['loss_classifier']/len(training_loader),
            'train_loss_box_reg': running_loss['loss_box_reg']/len(training_loader),
            'train_loss_objectness': running_loss['loss_objectness']/len(training_loader),
            'train_loss_rpn_box_reg': running_loss['loss_rpn_box_reg']/len(training_loader),
            'val_loss_classifier': val_loss['loss_classifier']/len(training_loader),
            'val_loss_box_reg': val_loss['loss_box_reg']/len(training_loader),
            'val_loss_objectness': val_loss['loss_objectness']/len(training_loader),
            'val_loss_rpn_box_reg': val_loss['loss_rpn_box_reg']/len(training_loader),
        }

        print(progress)

        with open('./object_detection/progress.csv', 'a', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=progress.keys(), delimiter=',')
            writer.writerow(progress)

        torch.save(state, param_path)
        if sum(val_loss.values()) < sum(state['best'].values()):
            state['best'] = val_loss
            state['model'] = net.state_dict()
            torch.save(state, "./object_detection/best_model.txt")

    print('Finished Training object detector')
