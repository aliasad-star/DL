import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
signal_groups = []  # To store signals in grouped form (30 groups)


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

    # Append the defected signal to the list
    all_signals.append(defected_signal_vector)

    # Read defect-free signal file
    defect_free_path = os.path.join(SIGNAL_FOLDER, f"{defect_free_file}.xlsx")
    defect_free_df = pd.read_excel(defect_free_path, header=0)

    # Extract the corresponding defect-free signal
    defect_free_signal_vector = defect_free_df.iloc[:, col_idx].values

    # Append the defect-free signal to the list
    all_defect_free_signals.append(defect_free_signal_vector)

# === NEW CODE BLOCK STARTS HERE === #

# Step 1: Group the signals
# Assume there are exactly 30 groups, each with 12 signals
for i in range(0, len(all_signals), GROUP_SIZE):
    group = {
        "signals": all_signals[i:i + GROUP_SIZE],  # Select 12 signals
        "defect_free_signals": all_defect_free_signals[i:i + GROUP_SIZE],  # Select 12 defect-free signals
        "coordinates": all_coord_matrices[i:i + GROUP_SIZE],  # Select corresponding coordinate matrices
        "bounding_boxes": all_defect_bboxes[i:i + GROUP_SIZE]  # Select corresponding bounding boxes
    }
    signal_groups.append(group)  # Append each group to the signal_groups list

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

