# Intellivision
ML Assignment 1 - Artificial Neural Networks

This project is multiple AI model that aim to classify the CIFAR10 dataset:

* A multi-layer perceptron model that has accuracy of 58+%
* A convolutional neural network model that has accuracy of 65+%
* A residual neural network model that has accuracy of 80+%

## Usage
`> python MODEL_NAME.py [-load | -save]`

`-load` load a models parameters and weights

`-save` trains the model and save the function parameters and weights

If no argument given, the model will simply train and show its accuracy

## Files
Files used in the implementation

### 1. utils.py
This contains most of the common utility functions shared by the models

### 2. MLP.py
This contains the implementation of the multi-layer perceptron model.

### 2. CNN.py
This contains the implementation of the convolutional neural network model.

### 2. RESNET.py
This contains the implementation of the residual neural network model.
