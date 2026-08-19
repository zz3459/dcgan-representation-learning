# DCGAN Unsupervised Representation Learning

TensorFlow implementation of DCGAN for unsupervised representation learning and downstream image classification.


This repository is a portfolio version of a team project completed
for Columbia University's E4040 course.

The materials highlighted in this repository focus on my individual
contributions to the project

## Overview

This project reproduces and extends experiments from
"Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks".

The DCGAN is trained on CelebA and Mini-ImageNet.
The discriminator is then reused as a frozen feature extractor
for downstream classification on CIFAR-10 and SVHN.

## My Contributions

- Implemented the 32×32 DCGAN Generator and Discriminator.
- Built data pipelines for Mini-ImageNet, CIFAR-10, and SVHN.
- Conducted Mini-ImageNet representation-learning experiments.
- Extracted discriminator features for CIFAR-10 and SVHN.
- Trained Linear L2-SVM classifiers for downstream evaluation.
- Contributed to experiment analysis and technical report writing.

## Pipeline

Mini-ImageNet
→ DCGAN Training
→ Frozen Discriminator
→ Feature Extraction
→ Linear L2-SVM
→ CIFAR-10 / SVHN Evaluation

## Results

| Dataset | Setting | Result |
|---|---|---|
| CIFAR-10 | Linear L2-SVM | 57% accuracy |
| SVHN | 1,000 labeled samples | ~40.8% test error |

## Tech Stack

Python · TensorFlow · Keras · NumPy · scikit-learn · DCGAN · SVM



**************************

# E4040 Fall 2025 Project  
## Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks

This project reproduces and analyzes key experiments from the paper  
**"Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (DCGAN)"**.

Our goal is to study whether DCGAN can learn meaningful hierarchical and semantic representations from unlabeled image data, whether these representations support interpretable latent-space manipulations, and whether the discriminator can learn reusable features for downstream classification tasks.

---

## Project Overview

We implement the DCGAN architecture and conduct experiments on multiple datasets. The project includes:

- Implementing DCGAN generator and discriminator architectures (`model.py`)
- Defining data loading and preprocessing functions (`data_utils.py`)
- Training DCGAN models on the LSUN dataset (`task1`)
- Training DCGAN models on the CelebA dataset and conducting face-related experiments (`task2`)
- Training DCGAN models on the mini-ImageNet dataset and conducting classification experiments using the discriminator (`task3`)

---

## Environment and Dependencies

This project is implemented using TensorFlow 2.4.0.  
All required packages and their versions are listed in `requirements.txt` to ensure reproducibility.

---

## Datasets

The experiments in this project use the following publicly available datasets:

- **LSUN**: A scene-centric dataset used for initial DCGAN training experiments. (task1)  
  The dataset is available at:  
  https://www.kaggle.com/datasets/jhoward/lsun_bedroom?select=data0

- **CelebA**: A large-scale face attributes dataset used for face generation and latent-space experiments. (task2)  
  The dataset can be downloaded from:  
  https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

- **mini-ImageNet**: A subset of the ImageNet dataset used for representation learning and classification experiments with the discriminator. (task3)   
  Mini-ImageNet can be obtained from:  
  https://www.kaggle.com/datasets/arjunashok33/miniimagenet?resource=download
  CIFAR10 can be obtained from:
  https://www.tensorflow.org/datasets/catalog/cifar10
  SVJN can be obtained from:
  https://www.tensorflow.org/datasets/catalog/svhn_cropped

In this project, the main experiments are conducted using the CelebA and mini-ImageNet datasets, corresponding to `task2` and `task3`, respectively. All experiments reported in this project are based on the implementations and results from these two tasks. The model training, experimental procedures, and result visualizations are all carried out within the corresponding Jupyter notebooks located in the `task2` and `task3` directories.


## Unsupervised Representation Learning with DCGAN: Generative Modeling and Discriminative Feature Evaluation
```

├── figures/ # Screenshots and figures for project submission
│ ├── gcp_work_example_screenshot_1.jpg
| ├── gcp_work_example_screenshot_2.jpg
| ├── gcp_work_example_screenshot_3.jpg
│ └── .DS_Store
│
├── task1/ # LSUN experiments
│ ├── data/
│ │ └── metadata/ # Dataset metadata
│ ├── experiments/ # Experiment scripts
│ ├── notebooks/ # Jupyter notebooks
│ ├── results/ # Experimental results
│ ├── ablation_runner.py # Ablation experiment runner
│ └── README.md # Task 1 documentation
│
├── task2/ # Face experiments (CelebA)
│ ├── celeba_training.ipynb # DCGAN training on CelebA
│ ├── generated_face/ # Selected generated face samples
│ ├── generated_faces/ # Images generated during training
│ ├── generated_samples_from_trained_model/
│ └── saved_models/ # Trained generator and discriminator models
│
├── task3/ # ImageNet experiments
│ ├── imagenet_feature_extraction.ipynb
│ ├── task3_images/ # Images generated during training
│ ├── generator_task3_imagenet.h5
│ ├── discriminator_task3_imagenet.h5
│ └── .DS_Store
│
├── model.py # DCGAN generator and discriminator definitions
├── train_dcgan.py # DCGAN training logic and loss functions
├── data_utils.py # Data loading and preprocessing utilities
├── requirements.txt # Python package dependencies
└── README.md # Project documentation

```
