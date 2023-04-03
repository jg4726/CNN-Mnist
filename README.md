# CNN-Mnist

This repository contains the Jupyter Notebook Cnn_Mnist.ipynb, which provides an implementation of a Convolutional Neural Network (CNN) for the task of handwritten digit recognition using the MNIST dataset. This code is designed for beginners who are interested in learning how to create and train deep learning models for image classification tasks.

# Table of Contents
1. Installation
2. Usage
3. Model Architecture
4. Dataset
5. Training and Evaluation

# Installation
1. Clone the repository
2. Launch Jupyter Notebook


# Usage
1. Open the Cnn_FCN.ipynb file in Jupyter Notebook.
2. Follow the instructions and comments provided in the notebook to understand each step of the process.
3. Execute each code cell in sequence by pressing Shift + Enter.
4. Modify the code as needed to adapt it to your own dataset or use case.

# Model Architecture
The implemented model consists of the following architecture:

1. A series of convolutional layers, each followed by ReLU activation and max-pooling.
2. One or more fully connected layers with dropout for regularization.
3. A softmax activation function to produce class probabilities.

# Dataset
This notebook assumes that you have an image dataset with labeled examples. The dataset should be organized into separate folders for each class, with images saved in a standard format (e.g., JPEG, PNG). The notebook provides code for loading and preprocessing the dataset, as well as splitting it into training and validation sets for model evaluation.

# Training and Evaluation
The notebook provides code for training the CNN-FCN model using stochastic gradient descent with a learning rate schedule. During training, the model's performance is evaluated using the validation set, and the best model weights are saved. After training, the notebook also provides code for evaluating the model's performance on the test set, as well as visualizing the confusion matrix and other metrics.
