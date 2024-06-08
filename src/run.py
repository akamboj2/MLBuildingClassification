"""
Runnable file.

Args:
    "--sample" number of sample to download from google street view
    "--fetch" number of samples to download from google images for the material classificator

    "--train-detector" start the training of the object detection
    "--train-segmentator" start the training of the object detection
    "--train-materials" start the training of the object detection

    "--test-detector" path to images to go through object detection
    "--test-segmentator" path to images to go through image segmentation
    "--test-materials" path to images to go through material clasification

    "--label-objects"  run object detection labeler
    "--label-segmentator" run segmentation labeler
    "--label-materials"  not implemented yet

    "--load-detector" loads the detector. Requires to have the key in the cofig file
    "--load-segmentator" loads the image segmentator. Requires to have the key in the cofig file
    "--load-materials" loads the material classifier. Requires to have the key in the cofig file

    "--save-detector" saves the object detector and saves the key
    "--save-segmentator" saves the object detector and saves the key
    "--save-materials" saves the material classifier and saves the key

    "--update" string of a dictionary of changes to perform to the configuration file

Example: run.py --fetch=1000 --train-material_classifier --load-material_classifier --save-material_classifier
"""
import sys

sys.path.insert(0, '')


import json
import argparse
import ast
from dataset import make_dataset as data
from image_segmentation import train as seg_train
from material_classifier import train as mat_train
from object_detection import train as dec_train

from material_classifier import fetch

from drive import load, save

from multiprocessing import Process
import os


def placeholder():
    """
    Placeholder function to ignore calls.
    """
    pass


def ensure_directories(config):
    """
    Ensures that the directories specified in the configuration

    :param config: configuration
    """
    if not os.path.isdir(root_path + config['object_detection']['labeled_dir']):
        raise Exception("Not found " + config['object_detection']['labeled_dir'] + " directory.")
    if not os.path.isdir(root_path + config['object_detection']['unlabeled_dir']):
        raise Exception("Not found " + config['object_detection']['unlabeled_dir'] + " directory.")

    if not os.path.isdir(root_path + config['image_segmentation']['labeled_dir']):
        raise Exception("Not found " + config['image_segmentation']['labeled_dir'] + " directory.")
    if not os.path.isdir(root_path + config['image_segmentation']['unlabeled_dir']):
        raise Exception("Not found " + config['image_segmentation']['unlabeled_dir'] + " directory.")

    if not os.path.isdir(root_path + config['material_classifier']['labeled_dir']):
        raise Exception("Not found " + config['material_classifier']['labeled_dir'] + " directory.")
    if not os.path.isdir(root_path + config['material_classifier']['unlabeled_dir']):
        raise Exception("Not found " + config['material_classifier']['unlabeled_dir'] + " directory.")


def read_config():
    """
    Reads the configuration file

    :return: the configuration dictionary
    """
    with open("./config.json", "r") as json_data:
        return json.load(json_data)


def update_config(updates):
    """
    Uppdates the entries of the dictionary

    :param updates: dictionary containing the new configuration
    :return: new configuration
    """
    with open("./config.json", "r") as json_data:
        data = json.load(json_data)

    for key in updates.keys():
        data[key] = updates[key]

    with open("./config.json", "w") as json_data:
        json.dump(data, json_data)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Building classifier"
    )

    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--fetch", type=int, default=0)

    parser.add_argument("--train-detector", action="store_true", default=False)
    parser.add_argument("--train-segmentator", action="store_true", default=False)
    parser.add_argument("--train-materials", action="store_true", default=False)

    parser.add_argument("--test-detector", type=str, default="")
    parser.add_argument("--test-segmentator", type=str, default="")
    parser.add_argument("--test-materials", type=str, default="")

    parser.add_argument("--label-objects", action="store_true", default=False)
    parser.add_argument("--label-segmentator", action="store_true", default=False)
    parser.add_argument("--label-materials", action="store_true", default=False)

    parser.add_argument("--load-detector", action="store_true", default=False)
    parser.add_argument("--load-segmentator", action="store_true", default=False)
    parser.add_argument("--load-materials", action="store_true", default=False)

    parser.add_argument("--save-detector", action="store_true", default=False)
    parser.add_argument("--save-segmentator", action="store_true", default=False)
    parser.add_argument("--save-materials", action="store_true", default=False)

    parser.add_argument("--update", type=str, default="")

    root_path = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
    args = parser.parse_args()
    sys.argv = [sys.argv[0]]
    config = None

    if args.update != "":
        data_to_update = ast.literal_eval(args.update)
        if not isinstance(data_to_update, dict):
            raise Exception("The updates were not given as a dictionary.")
        config = update_config(data_to_update)

    if config is None:
        config = read_config()

    ensure_directories(config)

    processes = []

    print("Commencing loading phase...")

    if args.load_detector:
        path, file = os.path.split(root_path + config['object_detection']['param_path'])
        p = Process(target=load, args=(
            config['object_detection']['key'],
            file,
            path
        ))
        p.start()
        processes.append(p)

    if args.load_materials:
        path, file = os.path.split(root_path + config['material_classification']['param_path'])
        p = Process(target=load, args=(
            config['material_classification']['key'],
            file,
            path
        ))
        p.start()
        processes.append(p)

    if args.load_segmentator:
        path, file = os.path.split(root_path + config['image_segmentation']['param_path'])
        p = Process(target=load, args=(
            config['image_segmentation']['key'],
            file,
            path
        ))
        p.start()
        processes.append(p)

    print("Commencing sampling phase...")

    if args.sample > 0:
        p = Process(target=data.img_from_db, args=(args.sample, 
            root_path + config['object_detection']['unlabeled_dir'],
            root_path))
        p.start()
        processes.append(p)

    if args.fetch > 0:
        p = Process(target=fetch.search_and_label, args=(
            config['material_classifier']['labels'],
            args.fetch,
            root_path + config['material_classifier']['labeled_dir'],
            root_path + config['material_classifier']['csv'],
        ))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("Commencing labeling phase...")

    if args.label_objects:
        from object_detection import labeler as dec_labeler
        dec_labeler.label(
            root_path + config['object_detection']['unlabeled_dir'],
            root_path + config['object_detection']['labeled_dir'],
            root_path + config['object_detection']['csv'],
            config['object_detection']['number_classes']
        )

    if args.label_segmentator:
        from image_segmentation import labeler as seg_labeler
        seg_labeler.label(
            root_path + config['image_segmentation']['unlabeled_dir'],
            root_path + config['image_segmentation']['labeled_dir'],
            root_path + config['image_segmentation']['csv'],
            config['image_segmentation']['number_classes']
        )

    for p in processes:
        p.join()

    print("Commencing training phase...")

    if args.train_detector:
        p = Process(target=dec_train.train_f_rcnn, args=(
            root_path + config['object_detection']['csv'],
            config['object_detection']['number_classes'],
            root_path + config['object_detection']['param_path'],
            config['object_detection']['img_size'],
            config['object_detection']['crop_size'],
            config['object_detection']['batch_size'],
            config['object_detection']['epochs'],
            config['object_detection']['lr'],
            root_path + config['object_detection']['labeled_dir']
        ))
        p.start()
        processes.append(p)

    if args.train_segmentator:
        p = Process(target=seg_train.train_segmentation, args=(
            root_path + config['image_segmentation']['csv'],
            root_path + config['image_segmentation']['labeled_dir'],
            config['image_segmentation']['number_classes'],
            root_path + config['image_segmentation']['param_path'],
            config['image_segmentation']['image_size'],
            config['image_segmentation']['crop_size'],
            config['image_segmentation']['batch_size'],
            config['image_segmentation']['epochs'],
            config['image_segmentation']['lr']
        ))
        p.start()
        processes.append(p)

    if args.train_materials:
        p = Process(target=mat_train.train_resnet, args=(
            root_path + config['material_classifier']['csv'],
            root_path + config['material_classifier']['labeled_dir'],
            config['material_classifier']['number_classes'],
            root_path + config['material_classifier']['param_path'],
            config['material_classifier']['image_size'],
            config['material_classifier']['crop_size'],
            config['material_classifier']['batch_size'],
            config['material_classifier']['epochs'],
            config['material_classifier']['lr']
        ))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("Commencing saving phase...")

    if args.save_detector:
        key = save(root_path + config['object_detection']['param_path'])
        config['object_detection']['key'] = key

    if args.save_materials:
        key = save(root_path + config['material_classifier']['param_path'])
        config['material_classifier']['key'] = key

    if args.save_segmentator:
        key = save(root_path + config['image_segmentation']['param_path'])
        config['image_segmentation']['key'] = key

    with open("./config.json", "w") as json_data:
        json.dump(config, json_data)

    print("Commencing testing phase...")

    if args.test_detector != "":
        r = dec_train.get_objects(
            args.test_detector,
            root_path + config['material_classifier']['param_path'],
            config['material_classifier']['number_classes'],
            config['material_classifier']['image_size']
        )
        print("The detection is :{}".format(r))

    if args.test_materials != "":
        r = mat_train.get_label(
            args.test_materials,
            root_path + config['material_classifier']['param_path'],
            config['material_classifier']['number_classes'],
            config['material_classifier']['image_size']
        )
        print("The material of the image is :{}".format(r))

    if args.test_segmentator != "":
        r = seg_train.get_label(
            args.test_segmentator,
            root_path + config['material_classifier']['param_path'],
            config['material_classifier']['number_classes'],
            config['material_classifier']['image_size']
        )
        print("The segmentation result is :{}".format(r))

    for p in processes:
        p.join()

    print("Closing the program...")


