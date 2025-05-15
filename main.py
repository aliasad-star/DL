import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Constants
PLATE_WIDTH = 400
PLATE_HEIGHT = 400
SIGNAL_FOLDER = "signals_data"  # Folder name where all signal files are stored

# Load master data
master_df = pd.read_excel("coords.xlsx")

# Prepare data containers
all_signals = []
all_coord_matrices = []
all_defect_bboxes = []

def normalize_coord(x, y):
    return x / (PLATE_WIDTH / 2), y / (PLATE_HEIGHT / 2)

def get_signal_index(tx_id, rx_id):
    if tx_id == rx_id:
        return None  # self-transmission is skipped
    # Receivers for a given transmitter (excluding self)
    valid_rxs = [i for i in range(1, 5) if i != tx_id]
    row_idx = tx_id - 1
    col_idx = valid_rxs.index(rx_id)
    return row_idx, col_idx

# Process each row in master file
for idx, row in master_df.iterrows():
    file_name   = row['File_Name']
    tx_id       = int(row['Transmitter_ID'])
    rx_id       = int(row['Receiver_ID'])

    #tx_id = int(tx_id_str.split('_')[1])
    #rx_id = int(rx_id_str.split('_')[1])

    tx_x, tx_y  = row['Tx_X'], row['Tx_Y']
    rx_x, rx_y  = row['Rx_X'], row['Rx_Y']
    defect_x, defect_y = row['Defect_X'], row['Defect_Y']
    defect_d    = row['Defect_Size']
    ...

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

    # Read signal file
    signal_path = os.path.join(SIGNAL_FOLDER, f"{file_name}.xlsx")
    signal_df = pd.read_excel(signal_path, header=0)

    # Debug info: which file, Tx-Rx pair
    #print(f"Reading signal from file: {file_name}, Tx: {tx_id}, Rx: {rx_id}")

    # Extract the signal using Tx-Rx pairing
    indices = get_signal_index(tx_id, rx_id)
    if indices is None:
        continue  # skip if Tx and Rx are same

    _, col_idx = indices
    signal_vector = signal_df.iloc[:, col_idx].values

    # Append signal
    all_signals.append(signal_vector)

# Final lists:
# - all_signals: [signal_1, signal_2, ...]
# - all_coord_matrices: [[[tx_x, tx_y], [rx_x, rx_y]], ...]
# - all_defect_bboxes: [[x_c, y_c, w, h], ...]

print(f"Loaded {len(all_signals)} signals.")

# Assuming you have file names list
for i in range(3):  # First 3 signals
    file_name = master_df.iloc[i]['File_Name']
    tx_id = master_df.iloc[i]['Transmitter_ID']  # e.g. 'T1'
    rx_id = master_df.iloc[i]['Receiver_ID']  # e.g. 'R2'
    header_label = f"{tx_id}-{rx_id}"

    print(f"Signal from file: {file_name}")
    print(f"Signal header: {header_label}")
    print(f"Signal Samples (first 20): {all_signals[i][:20]}")
    print(f"Signal Samples (last 20): {all_signals[i][-20:]}")
    print("=" * 50)

print (f"Shape of one signal vector: {all_signals[0].shape}")
#print (f"First few samples: {all_signals[0][:10]}")
print (f"Total coordinate pairs: {len(all_coord_matrices)}")
print (f"Total defect bounding boxes: {len(all_defect_bboxes)}")

#Preview first 3 items from each list
#print ("\nSamples signals: ")
#print(all_signals[:3])

print ("\nSample coordinate matrices:")
for coords in all_coord_matrices[:3]:
    print(coords)

print ("\nSamples defect bounding boxes:")
for bbox in all_defect_bboxes[:3]:
    print(bbox)