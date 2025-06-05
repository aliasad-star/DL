# --- SNIPPET: Import and Constants ---
import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fftpack import fft
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt

PLATE_WIDTH = 400
PLATE_HEIGHT = 400
SIGNAL_FOLDER = "signals_data"
GROUP_SIZE = 12
# Control training behaviour
skip_training = False   # Set to True to skip training and load model from file
# --- Load Metadata ---
master_df = pd.read_excel("coords.xlsx")

# --- Feature Extraction Functions ---
def extract_statistical_features(signal):
    return {
        "mean": np.mean(signal),
        "std": np.std(signal),
        "max": np.max(signal),
        "min": np.min(signal),
        "variance": np.var(signal),
        "skewness": skew(signal),
        "kurtosis": kurtosis(signal),
        "energy": np.sum(signal ** 2),
        "rms": np.sqrt(np.mean(signal ** 2))
    }

def extract_spectral_features(signal, sampling_rate=1.25e7):
    N = len(signal)
    fft_vals = np.abs(fft(signal))[:N // 2]
    fft_freqs = np.fft.fftfreq(N, d=1 / sampling_rate)[:N // 2]
    return {
        "spectral_energy": np.sum(fft_vals ** 2),
        "spectral_centroid": np.sum(fft_freqs * fft_vals) / np.sum(fft_vals),
        "spectral_bandwidth": np.sqrt(np.sum(((fft_freqs - np.sum(fft_freqs * fft_vals) / np.sum(fft_vals)) ** 2) * fft_vals) / np.sum(fft_vals)),
        "dominant_frequency": fft_freqs[np.argmax(fft_vals)]
    }

def extract_features(signal, sampling_rate=1.25e7):
    features = extract_statistical_features(signal)
    features.update(extract_spectral_features(signal, sampling_rate))
    return features

# --- Group Features ---
grouped_features, grouped_targets, grouped_filenames = [], [], []
unique_files = master_df['File_Name'].unique()

for file_name in unique_files:
    group_rows = master_df[master_df['File_Name'] == file_name]
    defect_x = group_rows['Defect_X'].iloc[0] / PLATE_WIDTH
    defect_y = group_rows['Defect_Y'].iloc[0] / PLATE_HEIGHT
    width_norm = group_rows['Defect_Size'].iloc[0] / PLATE_WIDTH
    height_norm = group_rows['Defect_Size'].iloc[0] / PLATE_HEIGHT
    #print(f"Defect Size: {group_rows['Defect_Size'].iloc[0]}, Normalized Width: {width_norm}, Height: {height_norm}, File: {file_name}")
    target = [defect_x, defect_y, width_norm, height_norm]

    signal_features = []
    for _, row in group_rows.iterrows():
        signal_file = os.path.join(SIGNAL_FOLDER, f"{row['File_Name']}.xlsx")
        signal_df = pd.read_excel(signal_file)
        signal = signal_df.iloc[:, 0].values
        features_array = np.array(list(extract_features(signal).values()))
        signal_features.append(features_array)

    concatenated_features = np.concatenate(signal_features)
    grouped_features.append(concatenated_features)
    grouped_targets.append(target)
    grouped_filenames.append(file_name)

features_df = pd.DataFrame(grouped_features)
targets_df = pd.DataFrame(grouped_targets, columns=["x", "y", "width", "height"])

# --- Correct Data Splitting Based on Group Filenames ---
filenames_train, filenames_temp = train_test_split(grouped_filenames, test_size=0.4, random_state=42)
filenames_val, filenames_test = train_test_split(filenames_temp, test_size=0.5, random_state=42)

# Print the splits
print("Training File Names:")
for group in filenames_train:
    print(group)

print("\nValidation File Names:")
for group in filenames_val:
    print(group)

print("\nTesting File Names:")
for group in filenames_test:
    print(group)

train_idx = [i for i, fname in enumerate(grouped_filenames) if fname in filenames_train]
val_idx = [i for i, fname in enumerate(grouped_filenames) if fname in filenames_val]
test_idx = [i for i, fname in enumerate(grouped_filenames) if fname in filenames_test]

train_set = set(filenames_train)
val_set = set(filenames_val)
test_set = set(filenames_test)

# Check for intersections between sets
train_val_overlap = train_set.intersection(val_set)
train_test_overlap = train_set.intersection(test_set)
val_test_overlap = val_set.intersection(test_set)

# Print results
print("\nOverlap between Training and Validation files:")
print(train_val_overlap if train_val_overlap else "No overlap found")

print("\nOverlap between Training and Test files:")
print(train_test_overlap if train_test_overlap else "No overlap found")

print("\nOverlap between Validation and Test files:")
print(val_test_overlap if val_test_overlap else "No overlap found")

X_train, y_train = features_df.iloc[train_idx], targets_df.iloc[train_idx]
X_val, y_val = features_df.iloc[val_idx], targets_df.iloc[val_idx]
X_test, y_test = features_df.iloc[test_idx], targets_df.iloc[test_idx]

# --- Train XGBoost Regressor ---
if not skip_training:
    print("\nTraining XGBoost Regressor...")
    xgboost_model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42
    )

    xgboost_model.fit(X_train, y_train)
    xgboost_model.save_model("xgboost_bbox_model.json")
else:
    print("\nSkipping training. Loading pre-trained model...")
    xgboost_model = xgb.XGBRegressor()
    xgboost_model.load_model("xgboost_bbox_model.json")

# --- Evaluation ---
y_val_pred = xgboost_model.predict(X_val)
val_mse = mean_squared_error(y_val, y_val_pred)
print(f"Validation MSE: {val_mse:.4f}")

y_test_pred = xgboost_model.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
print(f"Test MSE: {test_mse:.4f}")

# --- Display Bounding Boxes ---
def display_test_bounding_boxes(y_true, y_pred, filenames, num_samples=6):
    print("\nDisplaying True vs Predicted Bounding Boxes (Test Data):")
    for i in range(min(num_samples, len(y_true))):
        true_bbox = y_true.iloc[i].values
        pred_bbox = y_pred[i]
        print(f"File: {filenames[i]}")
        print(f"  Ground Truth: x={true_bbox[0]:.4f}, y={true_bbox[1]:.4f}, width={true_bbox[2]:.4f}, height={true_bbox[3]:.4f}")
        print(f"  Prediction:   x={pred_bbox[0]:.4f}, y={pred_bbox[1]:.4f}, width={pred_bbox[2]:.4f}, height={pred_bbox[3]:.4f}\n")

# Call with proper filenames
test_file_names = [grouped_filenames[i] for i in test_idx]
display_test_bounding_boxes(y_test, y_test_pred, test_file_names)
