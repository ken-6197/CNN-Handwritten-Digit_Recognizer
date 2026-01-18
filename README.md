Handwritten Digit Recognition using CNN (MNIST)

This project implements a Convolutional Neural Network (CNN) to recognize handwritten digits (0–9) using the MNIST dataset. The model is built using TensorFlow and Keras, trained on grayscale digit images of size 28×28, and achieves high accuracy in classifying handwritten digits. The project also supports testing with custom user-uploaded images in Google Colab.

The CNN architecture consists of multiple convolutional layers with ReLU activation, max pooling for feature reduction, and fully connected dense layers for classification. A softmax output layer with 10 neurons is used to predict the probability of each digit class.


Features

Trained on the MNIST handwritten digit dataset

Uses CNN architecture for better feature extraction

Achieves ~98–99% accuracy

Supports custom image testing

Implemented in Google Colab

Simple and beginner-friendly code


Technologies Used

Python

TensorFlow / Keras

NumPy

OpenCV

Matplotlib


Dataset

MNIST Dataset

60,000 training images

10,000 testing images

Image size: 28×28 grayscale


Model Architecture

Convolutional Layers (Conv2D)

ReLU Activation

MaxPooling

Flatten Layer

Fully Connected Dense Layers

Softmax Output Layer (10 classes)
