import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Constants
PLATE_WIDTH = 400
PLATE_HEIGHT = 400
SIGNAL_FOLDER = "signals_data"  # Folder name where all signal files are stored

# Load updated master data with new columns
master_df = pd.read_excel("coords.xlsx")

# Prepare data containers
all_signals = []
all_defect_free_signals = []  # To store defect-free signals
all_coord_matrices = []
all_defect_bboxes = []


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

    # Debug info: which file, Tx-Rx pair
    # print(f"Reading defected signal from file: {file_name}, Tx: {tx_id}, Rx: {rx_id}")

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

    # (Optional) Read and process time file if needed for additional steps
    # This assumes the time data may influence specific computations.
    # Uncomment the following code if you need to read and validate the shared time file.
    # time_df = pd.read_excel(time_file)
    # print(time_df.head())  # Just for debugging

# Final lists:
# - all_signals: [defected_signal_1, defected_signal_2, ...]
# - all_defect_free_signals: [defect_free_signal_1, defect_free_signal_2, ...]
# - all_coord_matrices: [[[tx_x, tx_y], [rx_x, rx_y]], ...]
# - all_defect_bboxes: [[x_c, y_c, w, h], ...]

print(f"Loaded {len(all_signals)} defected signals.")
print(f"Loaded {len(all_defect_free_signals)} defect-free signals.")

for i in range(3):  # First 3 signals
    file_name = master_df.iloc[i]['File_Name']
    tx_id = master_df.iloc[i]['Transmitter_ID']
    rx_id = master_df.iloc[i]['Receiver_ID']
    defect_free_file = master_df.iloc[i]['Defect_Free_File']
    header_label = f"{tx_id}-{rx_id}"

    print(f"Defected Signal: {file_name}, Defect-Free Signal: {defect_free_file}")
    print(f"Signal Header: {header_label}")
    print(f"Defected Signal Samples (first 20): {all_signals[i][:20]}")
    print(f"Defect-Free Signal Samples (first 20): {all_defect_free_signals[i][:20]}")
    print("=" * 50)

print(f"Total coordinate pairs: {len(all_coord_matrices)}")
print(f"Total defect bounding boxes: {len(all_defect_bboxes)}")

# Sample preview for debugging
print("\nSample coordinate matrices:")
for coords in all_coord_matrices[:3]:
    print(coords)

print("\nSample defect bounding boxes:")
for bbox in all_defect_bboxes[:3]:
    print(bbox)
