import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import skew, kurtosis
from scipy.fftpack import fft
from sklearn.model_selection import train_test_split  # For splitting dataset

import torch
from torch.utils.data import DataLoader, TensorDataset
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

        # Dynamically calculate the flattened size for fc1
        conv_out_length = self._calculate_conv_output_length(input_length)
        self.fc1 = nn.Linear(256 * conv_out_length, 128)  # Adjust dynamically
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, output_size)  # Output layer (sigmoid for normalized bounding box)

    def _calculate_conv_output_length(self, input_length):
        """Helper to calculate the output size after convolution and pooling layers."""
        length = input_length

        # Follow the exact sequence of layers to compute the output size
        length = (length - 4)  # conv1: kernel_size=5, stride=1
        length = (length - 2) // 2 + 1  # pool1: kernel_size=2, stride=2
        length = (length - 2)  # conv2: kernel_size=3, stride=1
        length = (length - 2) // 2 + 1  # pool2: kernel_size=2, stride=2
        length = (length - 2)  # conv3: kernel_size=3, stride=1

        return length

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Constants
PLATE_WIDTH = 400
PLATE_HEIGHT = 400
SIGNAL_FOLDER = "signals_data"  # Folder name where all signal files are stored
GROUP_SIZE = 12  # Each group contains 12 signals

# Load updated master data with new columns
master_df = pd.read_excel("coords.xlsx")

# Prepare data containers
all_signals = []  # To store defected signals
all_defect_free_signals = []  # To store defect-free signals
all_coord_matrices = []  # To store Tx-Rx coordinates
all_defect_bboxes = []  # To store bounding box labels
all_signal_features = []  # To store extracted features for defected signals
all_defect_free_signal_features = []  # To store extracted features for defect-free signals
signal_groups = []  # To store signals in grouped form (30 groups)


# Feature extraction functions
def extract_statistical_features(signal):
    """Extract key statistical features from a signal."""
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    max_val = np.max(signal)
    min_val = np.min(signal)
    variance_val = np.var(signal)
    skewness_val = skew(signal)
    kurtosis_val = kurtosis(signal)
    energy_val = np.sum(signal ** 2)
    rms_val = np.sqrt(np.mean(signal ** 2))

    return {
        "mean": mean_val,
        "std": std_val,
        "max": max_val,
        "min": min_val,
        "variance": variance_val,
        "skewness": skewness_val,
        "kurtosis": kurtosis_val,
        "energy": energy_val,
        "rms": rms_val
    }


def extract_spectral_features(signal, sampling_rate=1):
    """Extract frequency-domain features from a signal using FFT."""
    N = len(signal)
    fft_vals = np.abs(fft(signal))  # Magnitude of FFT
    fft_freqs = np.fft.fftfreq(N, d=1 / sampling_rate)  # FFT frequencies

    # Retain only the positive half of frequencies
    fft_vals = fft_vals[:N // 2]
    fft_freqs = fft_freqs[:N // 2]

    # Compute spectral features
    total_energy = np.sum(fft_vals ** 2)
    spectral_centroid = np.sum(fft_freqs * fft_vals) / np.sum(fft_vals)
    spectral_bandwidth = np.sqrt(np.sum(((fft_freqs - spectral_centroid) ** 2) * fft_vals) / np.sum(fft_vals))
    dominant_frequency = fft_freqs[np.argmax(fft_vals)]

    return {
        "spectral_energy": total_energy,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "dominant_frequency": dominant_frequency
    }


def extract_features(signal, sampling_rate=1):
    """Combine statistical and spectral features into a single feature dictionary."""
    features = {}
    features.update(extract_statistical_features(signal))
    features.update(extract_spectral_features(signal, sampling_rate))
    return features


def normalize_coord(x, y):
    """Normalize coordinates to range [-1, 1] for both x and y."""
    return x / (PLATE_WIDTH / 2), y / (PLATE_HEIGHT / 2)


def get_signal_index(tx_id, rx_id):
    """Retrieve signal matrix index based on Transmitter-Receiver IDs."""
    if tx_id == rx_id:
        return None  # self-transmission is skipped
    # Receivers for a given transmitter (excluding self)
    valid_rxs = [i for i in range(1, 5) if i != tx_id]
    row_idx = tx_id - 1
    col_idx = valid_rxs.index(rx_id)
    return row_idx, col_idx


# Process each row in updated master file
for idx, row in master_df.iterrows():
    # Extract necessary columns
    file_name = row['File_Name']  # Defected signal file
    defect_free_file = row['Defect_Free_File']  # Corresponding defect-free signal file
    time_file = row['Time_File']  # Shared time data reference

    tx_id = int(row['Transmitter_ID'])
    rx_id = int(row['Receiver_ID'])
    tx_x, tx_y = row['Tx_X'], row['Tx_Y']
    rx_x, rx_y = row['Rx_X'], row['Rx_Y']
    defect_x, defect_y = row['Defect_X'], row['Defect_Y']
    defect_d = row['Defect_Size']

    # Normalize coordinates
    tx_x_norm, tx_y_norm = normalize_coord(tx_x, tx_y)
    rx_x_norm, rx_y_norm = normalize_coord(rx_x, rx_y)

    # Add coordinate matrix for Transmitter (Tx) and Receiver (Rx)
    all_coord_matrices.append([[tx_x_norm, tx_y_norm], [rx_x_norm, rx_y_norm]])

    # Normalize defect-free bounding box
    bbox_x_norm = row['Defect_Free_X'] / PLATE_WIDTH
    bbox_y_norm = row['Defect_Free_Y'] / PLATE_HEIGHT
    bbox_w_norm = row['Defect_Free_W'] / PLATE_WIDTH
    bbox_h_norm = row['Defect_Free_H'] / PLATE_HEIGHT

    # Create normalized bounding boxes for defects
    width_norm = defect_d / PLATE_WIDTH
    height_norm = defect_d / PLATE_HEIGHT
    defect_x_norm = defect_x / PLATE_WIDTH
    defect_y_norm = defect_y / PLATE_HEIGHT
    defect_bbox = [defect_x_norm, defect_y_norm, width_norm, height_norm]

    # Add both normalized bounding boxes to the list
    # (You can decide which one is relevant during training)
    all_defect_bboxes.append(defect_bbox)

    # Debug output for validation
    print(f"Defected BBox (Normalized): {defect_bbox}")
    print(f"Defect-Free BBox (Normalized): [{bbox_x_norm}, {bbox_y_norm}, {bbox_w_norm}, {bbox_h_norm}]")

    # Read defected signal file
    signal_path = os.path.join(SIGNAL_FOLDER, f"{file_name}.xlsx")
    signal_df = pd.read_excel(signal_path, header=0)

    # Extract the defected signal using Tx-Rx pairing
    indices = get_signal_index(tx_id, rx_id)
    if indices is None:
        continue  # Skip if Tx and Rx are the same

    _, col_idx = indices
    defected_signal_vector = signal_df.iloc[:, col_idx].values

    # Append the defected signal and extract features
    all_signals.append(defected_signal_vector)
    all_signal_features.append(extract_features(defected_signal_vector))

    # Read defect-free signal file
    defect_free_path = os.path.join(SIGNAL_FOLDER, f"{defect_free_file}.xlsx")
    defect_free_df = pd.read_excel(defect_free_path, header=0)

    # Extract the corresponding defect-free signal
    defect_free_signal_vector = defect_free_df.iloc[:, col_idx].values

    # Append the defect-free signal and extract features
    all_defect_free_signals.append(defect_free_signal_vector)
    all_defect_free_signal_features.append(extract_features(defect_free_signal_vector))

# Step 1: Group the signals
for i in range(0, len(all_signals), GROUP_SIZE):
    group_file_names = master_df['File_Name'][i:i + GROUP_SIZE].tolist()

    group = {
        "file_names": group_file_names,
        "signals": all_signals[i:i + GROUP_SIZE],
        "defect_free_signals": all_defect_free_signals[i:i + GROUP_SIZE],
        "coordinates": all_coord_matrices[i:i + GROUP_SIZE],
        "bounding_boxes": all_defect_bboxes[i:i + GROUP_SIZE],
        "features": all_signal_features[i:i + GROUP_SIZE],  # Include features in the group
        "defect_free_features": all_defect_free_signal_features[i:i + GROUP_SIZE]  # Include defect-free features
    }
    signal_groups.append(group)

print(f"Total groups formed: {len(signal_groups)} (Each group has {GROUP_SIZE} signals)")

# Step 2: Split the data into train, validation, and test sets
# For the splitting, we focus on maintaining the grouping
train_groups, temp_groups = train_test_split(signal_groups, test_size=0.4, random_state=42)  # 60% Train
val_groups, test_groups = train_test_split(temp_groups, test_size=0.5, random_state=42)  # 20% Validation, 20% Test

print("\nTesting File Names:")
for group in test_groups:
    print(group["file_names"])

print(f"Training Groups: {len(train_groups)}")
print(f"Validation Groups: {len(val_groups)}")
print(f"Testing Groups: {len(test_groups)}")

# Step 3: Prepare the flattened datasets for training
# Flatten grouped data into lists for easier processing by the model

def flatten_groups(groups):
    """
    Flatten grouped data containing defected and defect-free signals.

    Parameters:
        groups: list
            Groups of signals with both defected and defect-free data, along with bounding boxes.

    Returns:
        signals: np.array
            Flattened defected signal data.
        defect_free_signals: np.array
            Flattened defect-free signal data.
        bboxes: np.array
            Flattened bounding boxes for defected signals.
    """
    signals = []
    defect_free_signals = []
    bboxes = []

    for group in groups:
        # Collect defected signals and their bounding boxes
        signals.extend(group["signals"])
        bboxes.extend(group["bounding_boxes"])

        # Collect defect-free signals
        defect_free_signals.extend(group["defect_free_signals"])

    # Return the flattened arrays
    return np.array(signals), np.array(defect_free_signals), np.array(bboxes)

# Flatten training data
X_train_def, X_train_def_free, y_train = flatten_groups(train_groups)

# Flatten validation data
X_val_def, X_val_def_free, y_val = flatten_groups(val_groups)

# Flatten testing data
X_test_def, X_test_def_free, y_test = flatten_groups(test_groups)

# Print shapes for verification
print(f"Defected Train Data: {X_train_def.shape}, Bounding Boxes: {y_train.shape}")
print(f"Defect-Free Train Data: {X_train_def_free.shape}")
print(f"Defected Validation Data: {X_val_def.shape}, Bounding Boxes: {y_val.shape}")
print(f"Defect-Free Validation Data: {X_val_def_free.shape}")
print(f"Defected Test Data: {X_test_def.shape}, Bounding Boxes: {y_test.shape}")
print(f"Defect-Free Test Data: {X_test_def_free.shape}")

# Combine defected and defect-free signals for training
X_train_combined = np.concatenate([X_train_def, X_train_def_free], axis=0)
y_train_combined = np.concatenate([y_train, np.zeros((X_train_def_free.shape[0], 4))], axis=0)

# Combine defected and defect-free signals for validation
X_val_combined = np.concatenate([X_val_def, X_val_def_free], axis=0)
y_val_combined = np.concatenate([y_val, np.zeros((X_val_def_free.shape[0], 4))], axis=0)

# Combine defected and defect-free signals for testing
X_test_combined = np.concatenate([X_test_def, X_test_def_free], axis=0)
y_test_combined = np.concatenate([y_test, np.zeros((X_test_def_free.shape[0], 4))], axis=0)

print(f"y_test shape: {y_test.shape}")
print(f"Defect-Free Shape for Zeros: {np.zeros((X_test_def_free.shape[0], 4)).shape}")
print(f"y_test_combined shape: {y_test_combined.shape}")

# Verify the combined shapes
print(f"Combined Train Data: {X_train_combined.shape}, Labels: {y_train_combined.shape}")
print(f"Combined Validation Data: {X_val_combined.shape}, Labels: {y_val_combined.shape}")
print(f"Combined Test Data: {X_test_combined.shape}, Labels: {y_test_combined.shape}")

# Reshape signals to add a dimension for features (channels) for compatibility with CNN
X_train = X_train_combined.reshape(X_train_combined.shape[0], 1, X_train_combined.shape[1])  # Shape: (216, 1, 2501)
X_val = X_val_combined.reshape(X_val_combined.shape[0], 1, X_val_combined.shape[1])  # Shape: (validation_size, 1, sequence_length)
X_test = X_test_combined.reshape(X_test_combined.shape[0], 1, X_test_combined.shape[1])  # Shape: (test_size, 1, sequence_length)

# Verify reshaped data shapes
print(f"Reshaped X_train shape: {X_train.shape}")  # Should print: (216, 1, 2501)
print(f"Reshaped X_val shape: {X_val.shape}")  # Should be: (validation_size, 1, sequence_length)
print(f"Reshaped X_test shape: {X_test.shape}")  # Should be: (test_size, 1, sequence_length)

# Debug: Verify the combined data shapes before proceeding
print(f"Shape of X_test_combined: {X_test_combined.shape}")  # Should match (144, 2501)
print(f"Shape of y_test_combined: {y_test_combined.shape}")  # Should match (144, 4)

# Debug: Verify reshaped X_test shape
print(f"Shape of reshaped X_test: {X_test.shape}")  # Should match (144, 1, 2501)
# Detect the computation device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Convert test data to tensors
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test_combined, dtype=torch.float32).to(device)

# Debug: Check tensor shapes for the dataset
print(f"Shape of X_test_tensor: {X_test_tensor.shape}")  # (144, 1, 2501)
print(f"Shape of y_test_tensor: {y_test_tensor.shape}")  # (144, 4)
# Create the test dataset
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Update input_length and num_features for your reshaped input
num_features = X_train.shape[1]  # Now equals 1 (number of features per time step)
input_length = X_train.shape[2]  # Sequence length (2501)
output_size = y_train.shape[1]
# Instantiate the CNN model

model = CNN1D(input_length=input_length, num_features=num_features, output_size=output_size)

# Move the model to a GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#import torch
#from torch.utils.data import DataLoader, TensorDataset

# Assuming X_train, X_val, X_test, y_train, y_val, y_test are NumPy arrays
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# Convert test data to tensors
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
# Create a Dataset for testing
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 16
# Create DataLoader for the test set
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Verify test DataLoader
print(f"Test DataLoader created with {len(test_dataset)} samples.")

# Create Datasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
print(f"Shape of X_test_tensor: {X_test_tensor.shape}")
print(f"Shape of y_test_tensor: {y_test_tensor.shape}")
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create DataLoaders for batching and shuffling
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Train DataLoader created with {len(train_dataset)} samples.")
print(f"Validation DataLoader created with {len(val_dataset)} samples.")
print(f"Test DataLoader created with {len(test_dataset)} samples.")

# Debugging: Test the model with dummy input
model.eval()  # Set model to evaluation mode for testing
with torch.no_grad():  # Disable gradient computation
    dummy_input = torch.randn(16, 1, input_length).to(device)  # Batch of 16 signals, 1 feature, input_length
    dummy_output = model(dummy_input)  # Pass dummy input through the model
    print(f"Dummy input shape: {dummy_input.shape}")  # Should be (16, 1, input_length)
    print(f"Model output shape (dummy): {dummy_output.shape}")  # Should match (16, output_size)

# Print model summary
print(model)

if __name__ == "__main__":
    # Add this flag to control whether training is skipped
    skip_training = False  # Set to True to skip training and load pre-trained weights

    # Model parameters
    input_length = X_train.shape[2]  # Length of each signal
    num_features = X_train.shape[1]  # Number of input features (channels)
    output_size = y_train.shape[1]  # Output size (bounding box dimensions)

    # Instantiate the model
    model = CNN1D(input_length=input_length, num_features=num_features, output_size=output_size)
    model.to(device)  # Move to device (CPU or GPU)

    if not skip_training:
        # Training phase
        print("Starting model training...")

        # Define the loss function
        criterion = nn.SmoothL1Loss() #Better for bounding box regression

        # Define the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Adam optimizer

        # Training loop
        num_epochs = 50  # Number of epochs

        for epoch in range(num_epochs):
            model.train()  # Set the model in training mode
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                # Move data to the same device as the model
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                # Forward pass
                outputs = model(X_batch)  # Get model predictions

                print(f"Input batch shape: {X_batch.shape}")  # Shape of input to the model
                print(f"Model output shape: {outputs.shape}")  # Shape of the model's predictions
                print(f"Bounding box shape: {y_batch.shape}")  # Shape of the target/ground truth

                loss = criterion(outputs, y_batch)  # Compute loss

                # Backward pass
                optimizer.zero_grad()  # Clear gradients
                loss.backward()  # Backpropagation of loss
                optimizer.step()  # Update model parameters

                train_loss += loss.item()  # Accumulate training loss

            # Validation
            model.eval()  # Set the model to evaluation mode
            val_loss = 0.0
            with torch.no_grad():  # Disable gradient computation for validation
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()

            # Print progress
            print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss / len(train_loader):.4f}, "
                  f"Validation Loss: {val_loss / len(val_loader):.4f}")


        # Evaluate the model
        model.eval()
        with torch.no_grad():  # Disable gradient computation
            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor)

        print(f"Test Loss: {test_loss.item():.4f}")

        # Save the trained model
        torch.save(model.state_dict(), "cnn_bbox_model.pth")
    else:
        # Loading phase
        print("Skipping training. Loading the pre-trained model...")
        # Verify if `cnn_bbox_model.pth` exists before loading
        try:
            model.load_state_dict(torch.load("cnn_bbox_model.pth"))
            print("Pre-trained model loaded successfully!")
        except FileNotFoundError:
            print("Error: cnn_bbox_model.pth file not found. Please train the model first!")
            exit(1)  # Exit the script if the model file is missing

    # Inference phase (common to both training and skipping training)
    print("Running inference on the test dataset...")

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculations for inference
        all_predictions = []
        all_ground_truth = []
        group_predictions = {}
        group_ground_truths = {}

        group_idx = 0  # To track which group the batch belongs to
        GROUP_SIZE = 12  # Assuming 12 signals per group

        for batch_idx, batch in enumerate(test_loader):  # Enumerate for tracking batch index
            X_batch, y_batch = batch
            X_batch = X_batch.to(device)

            # Get predictions
            predictions = model(X_batch)
            predictions = predictions.clamp(-1, 1)

            # Collect predictions and ground truths
            all_predictions.extend(predictions.cpu().numpy())  # Store all predictions
            all_ground_truth.extend(y_batch.cpu().numpy())  # Store all ground truths

            # Assign predictions and ground truth to current group
            # The assumption here is that batches align perfectly with group sizes
            for i in range(len(X_batch)):
                current_group_index = group_idx // GROUP_SIZE + 1
                group_name = f"Group_{current_group_index}"

                if group_name not in group_predictions:
                    group_predictions[group_name] = []
                    group_ground_truths[group_name] = []

                group_predictions[group_name].append(predictions[i].cpu().numpy())
                group_ground_truths[group_name].append(y_batch[i].cpu().numpy())

                group_idx += 1

            # Debug: Print shape and examples
            print(f"Batch {batch_idx + 1}:")
            print(f"  Prediction shape: {predictions.shape}")
            print(f"  Ground truth shape: {y_batch.shape}")
            print(f"  Sample predictions: {predictions[0]}")
            print(f"  Sample ground truth: {y_batch[0]}")

    # Debugging and verification
    #print(f"\nCollected results for {len(group_predictions)} groups.")
    #for group, preds in group_predictions.items():
        #print(f"  {group}: {len(preds)} samples")

    # Visualize the first test sample
    signal = X_test[0][0] # Replace with your signal data slice
    ground_truth_bbox = all_ground_truth[0]  # Replace with ground truth
    predicted_bbox = all_predictions[0]  # Replace with model's prediction

    #print(f"Signal shape: {signal.shape}, Signal sample: {signal[:10]}")
    print(f"Test File: {file_name}")
    print(f"Ground truth BBox: {ground_truth_bbox}")
    print(f"Predicted BBox: {predicted_bbox}")

def plot_signal_with_predictions(signal, ground_truth, predicted):
    """
    Visualizes the input signal and overlays the ground truth
    and predicted bounding boxes.
    """
    # Debug signal information
    print(f"Signal length: {len(signal)}")
    print(f"Signal sample (first 10 values): {signal[:10]}")

    # Debug bounding box information
    x_gt, y_gt, w_gt, h_gt = ground_truth
    x_pred, y_pred, w_pred, h_pred = predicted
    print(f"Ground Truth BBox: x={x_gt}, y={y_gt}, width={w_gt}, height={h_gt}")
    print(f"Predicted BBox: x={x_pred}, y={y_pred}, width={w_pred}, height={h_pred}")

    # If bounding boxes are normalized, rescale to signal length
    signal_length = len(signal)
    if abs(x_gt) <= 1 and abs(w_gt) <= 1:  # Assuming normalized
        x_gt = int(x_gt * signal_length)
        w_gt = int(w_gt * signal_length)

    if abs(x_pred) <= 1 and abs(w_pred) <= 1:  # Assuming normalized
        x_pred = int(x_pred * signal_length)
        w_pred = int(w_pred * signal_length)

    print(f"Rescaled Ground Truth BBox: x={x_gt}, width={w_gt}")
    print(f"Rescaled Predicted BBox: x={x_pred}, width={w_pred}")

    plt.figure(figsize=(10, 6))
    plt.plot(signal, label='Signal', color='blue')

    # Plot ground truth
    plt.axvspan(x_gt, x_gt + w_gt, color='green', alpha=0.3, label='Ground Truth')

    # Plot prediction
    plt.axvspan(x_pred, x_pred + w_pred, color='red', alpha=0.3, label='Prediction')

    # Title and labels
    plt.title("Signal with Bounding Boxes")
    plt.xlabel("Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.savefig('signal_with_predictions.png')
   # plt.show()