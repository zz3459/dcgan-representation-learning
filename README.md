# DCGAN Unsupervised Representation Learning

This project implements a 32×32 Deep Convolutional Generative Adversarial Network (DCGAN) in TensorFlow and evaluates the learned discriminator representations on downstream image-classification tasks.
This repository is a portfolio version of a team course project at Columbia University. The code and materials included here focus on my individual contributions.
## Project Overview

The model is trained on Mini-ImageNet using unsupervised adversarial learning. After training, the discriminator is frozen and used as a feature extractor for CIFAR-10 and SVHN.

A Linear L2-SVM is trained on the extracted convolutional features to evaluate the transferability of the learned representations.

## My Contributions

- Implemented the 32×32 DCGAN Generator and Discriminator.
- Built data-loading and preprocessing pipelines for Mini-ImageNet, CIFAR-10, and SVHN.
- Implemented the Mini-ImageNet training and downstream feature-extraction pipeline.
- Evaluated learned representations using Linear L2-SVM classification on CIFAR-10 and SVHN.
- Contributed to experiment analysis and technical report writing.

## Results

- CIFAR-10 classification accuracy: 57%
- SVHN test error: approximately 40.8% using only 1,000 labeled training samples.

## Technologies

- Python
- TensorFlow / Keras
- NumPy
- scikit-learn
- DCGAN
- Linear SVM
- Unsupervised Representation Learning
