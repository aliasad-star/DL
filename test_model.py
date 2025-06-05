import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.fftpack import fft
import xgboost as xgb  # Install if not already done: pip install xgboost

# Constants
PLATE_WIDTH = 400
PLATE_HEIGHT = 400
GROUP_SIZE = 12
SAMPLING_RATE = 1.25e7


# Feature Extraction Functions
def extract_statistical_features(signal):
    """Extract statistical features from a signal."""
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


def extract_spectral_features(signal, sampling_rate=SAMPLING_RATE):
    """Extract frequency-domain features from signals."""
    N = len(signal)
    fft_vals = np.abs(fft(signal))
    fft_freqs = np.fft.fftfreq(N, d=1 / sampling_rate)

    fft_vals = fft_vals[:N // 2]
    fft_freqs = fft_freqs[:N // 2]

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


def extract_features(signal, sampling_rate=SAMPLING_RATE):
    features = {}
    features.update(extract_statistical_features(signal))
    features.update(extract_spectral_features(signal, sampling_rate))
    return features


def prepare_test_data(metadata_file, signal_folder, group_size):
    """
    Prepare data for testing with the metadata file and signal folder.
    """
    # Load the metadata file
    metadata_df = pd.read_excel(metadata_file)

    # Containers for data
    all_signal_features = []  # Extracted features for defected signals
    all_filenames = []

    # Process metadata
    for _, row in metadata_df.iterrows():
        file_name = row["File_Name"]
        signal_path = os.path.join(signal_folder, f"{file_name}.xlsx")

        # Load the signal data
        try:
            signal_df = pd.read_excel(signal_path, header=0)
        except FileNotFoundError:
            print(f"Warning: File {file_name}.xlsx not found in {signal_folder}. Skipping...")
            continue

        # Assume first column contains the signal
        defected_signal = signal_df.iloc[:, 0].values

        # Extract features
        features_dict = extract_features(defected_signal)
        features_array = np.array(list(features_dict.values()))  # Convert to array
        all_signal_features.append(features_array)
        all_filenames.append(file_name)

    # Group features (12 signals per defect)
    grouped_features = []
    grouped_filenames = []

    for i in range(0, len(all_signal_features), group_size):
        group_signals = all_signal_features[i:i + group_size]
        if len(group_signals) == group_size:  # Ensure complete groups
            concatenated_features = np.concatenate(group_signals)
            grouped_features.append(concatenated_features)
            grouped_filenames.append(all_filenames[i])

    # Convert features to a DataFrame
    features_df = pd.DataFrame(grouped_features)

    return features_df, grouped_filenames


def predict_and_display(model_path, test_features, test_filenames):
    """
    Load the pre-trained model, make predictions, and display results.
    """
    # Load the trained model
    print("\nLoading the pre-trained model...")
    xgboost_model = xgb.XGBRegressor()
    xgboost_model.load_model(model_path)
    print("Model loaded successfully.")

    # Make predictions
    print("\nMaking predictions on the test dataset...")
    predictions = xgboost_model.predict(test_features)

    # Display results
    print(f"\nDisplaying predictions for {len(test_filenames)} defects:")
    for i, filename in enumerate(test_filenames):
        bbox = predictions[i]
        print(f"File: {filename}")
        print(f"  Predicted Bounding Box: x={bbox[0]:.4f}, y={bbox[1]:.4f}, "
              f"width={bbox[2]:.4f}, height={bbox[3]:.4f}")


# Main Entry Point
if __name__ == "__main__":
    print("Preparing data for testing...")

    # Paths
    metadata_file = "coords.xlsx"  # Path to new test metadata
    signal_folder = "signals_data"  # Path to folder containing signals
    model_path = "xgboost_bbox_model.json"  # Path to pre-trained model

    # Prepare the test data
    test_features, test_filenames = prepare_test_data(
        metadata_file=metadata_file,
        signal_folder=signal_folder,
        group_size=GROUP_SIZE
    )

    # Ensure test data is not empty
    if len(test_features) > 0:
        # Run predictions and display results
        predict_and_display(model_path, test_features, test_filenames)
    else:
        print("No test data prepared. Ensure metadata and signal files are correct.")
