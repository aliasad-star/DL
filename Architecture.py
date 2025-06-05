import tensorflow as tf
from tensorflow.keras import layers, models


# Define 1D CNN Model for Bounding Box Prediction
def create_1d_cnn(input_length, num_features, output_size):
    model = models.Sequential()
    model.add(layers.Conv1D(64, kernel_size=5, activation='relu', input_shape=(input_length, num_features)))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(128, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(256, kernel_size=3, activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(output_size, activation='sigmoid'))  # Bounding box normalized outputs
    return model

## Pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F


# Define 1D CNN Model for Bounding Box Prediction
class CNN1D(nn.Module):
    def __init__(self, input_length, num_features, output_size):
        super(CNN1D, self).__init__()
        # Define the layers of your 1D CNN
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=5)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * ((input_length - 4) // 2 - 2) // 2,
                             128)  # Calculate the final size after Conv1D & pooling
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, output_size)  # Output layer (sigmoid for normalized bounding box)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))  # Sigmoid activation to normalize outputs (optional)
        return x
