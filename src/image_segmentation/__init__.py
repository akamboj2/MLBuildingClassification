"""
Trains a image classifier using the deeplabv3 resnet50 model.

The input to the model is expected to be a list of tensors, each of shape [C, H, W].

It uses CrossEntropyLoss as loss function and Adam as optimizer.

During training it expects every image to have a pixel-wise label between 0 and the number of classes - 1.
"""