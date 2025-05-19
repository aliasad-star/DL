import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import skew, kurtosis
from scipy.fftpack import fft
from sklearn.model_selection import train_test_split  # For splitting dataset

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

    # Add coordinate matrix
    all_coord_matrices.append([[tx_x_norm, tx_y_norm], [rx_x_norm, rx_y_norm]])

    # Create normalized bounding box for circular defect
    width_norm = defect_d / PLATE_WIDTH
    height_norm = defect_d / PLATE_HEIGHT
    defect_x_norm = defect_x / PLATE_WIDTH
    defect_y_norm = defect_y / PLATE_HEIGHT
    defect_bbox = [defect_x_norm, defect_y_norm, width_norm, height_norm]
    all_defect_bboxes.append(defect_bbox)

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
    group = {
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

print(f"Training Groups: {len(train_groups)}")
print(f"Validation Groups: {len(val_groups)}")
print(f"Testing Groups: {len(test_groups)}")


# Step 3: Prepare the flattened datasets for training
# Flatten grouped data into lists for easier processing by the model
def flatten_groups(groups):
    signals = []
    defect_free_signals = []
    bboxes = []
    for group in groups:
        signals.extend(group["signals"])
        defect_free_signals.extend(group["defect_free_signals"])
        bboxes.extend(group["bounding_boxes"])
    return np.array(signals), np.array(defect_free_signals), np.array(bboxes)


# Flatten training, validation, and testing groups
X_train, X_train_def_free, y_train = flatten_groups(train_groups)
X_val, X_val_def_free, y_val = flatten_groups(val_groups)
X_test, X_test_def_free, y_test = flatten_groups(test_groups)

print(f"Training Data: Signals - {X_train.shape}, Bounding Boxes - {y_train.shape}")
print(f"Validation Data: Signals - {X_val.shape}, Bounding Boxes - {y_val.shape}")
print(f"Testing Data: Signals - {X_test.shape}, Bounding Boxes - {y_test.shape}")
