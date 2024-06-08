"""
Trains a image classifier using a resnet model.

The input to the model is expected to be a list of tensors, each of shape [C, H, W].

It uses CrossEntropyLoss as loss function and Adam as optimizer.

During training it expects every image to have a label between 0 and the number of classes - 1.
"""