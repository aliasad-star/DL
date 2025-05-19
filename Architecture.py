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
