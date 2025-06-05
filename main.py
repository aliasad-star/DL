import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.stats import zscore

# --- CONFIGURATION ---
SIGNAL_FOLDER = "signals_data"
COORDS_FILE = "coords.xlsx"
SIGNAL_LENGTH = 2500
NUM_CHANNELS = 12
PLATE_WIDTH = 400
PLATE_HEIGHT = 400
BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- DATASET CLASS ---
class SignalDataset(Dataset):
    def __init__(self, filenames, coords_df):
        self.filenames = filenames
        self.coords_df = coords_df

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        group_name = self.filenames[idx]
        group_rows = self.coords_df[self.coords_df['File_Name'] == group_name]

        # Load signals and normalize each with z-score
        signals = []
        signal_file = os.path.join(SIGNAL_FOLDER, f"{group_name}.xlsx")
        signal_df = pd.read_excel(signal_file)

        # --- HELPER FUNCTION ---
        def safe_zscore(signal):
            std = np.std(signal)
            if std == 0 or np.isnan(std):
                return np.zeros_like(signal)
            return (signal - np.mean(signal)) / std

        for i in range(NUM_CHANNELS):
            signal = signal_df.iloc[i, :SIGNAL_LENGTH].values.astype(np.float32)
            signal = safe_zscore(signal)  # normalize
            signals.append(signal)

        signal_tensor = torch.tensor(np.array(signals), dtype=torch.float32)

        # Normalize target
        defect_x = group_rows['Defect_X'].iloc[0] / PLATE_WIDTH
        defect_y = group_rows['Defect_Y'].iloc[0] / PLATE_HEIGHT
        width_norm = group_rows['Defect_Size'].iloc[0] / PLATE_WIDTH
        height_norm = group_rows['Defect_Size'].iloc[0] / PLATE_HEIGHT
        target = torch.tensor([defect_x, defect_y, width_norm, height_norm], dtype=torch.float32)

        return signal_tensor, target


# --- MODEL ---
class DefectCNN(nn.Module):
    def __init__(self):
        super(DefectCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),

            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.net(x)


# --- TRAINING UTILITIES ---
def train_model(model, train_loader, val_loader, optimizer, criterion):
    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_defect_cnn.pt")


# --- EVALUATION AND VISUALIZATION ---
def evaluate_and_plot(model, test_loader, test_filenames, coords_df):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            preds = model(xb).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(yb.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    mse = mean_squared_error(all_targets, all_preds)
    print(f"\nTest MSE: {mse:.4f}\n")

    # Plot true vs predicted bounding boxes
    for i in range(min(6, len(all_preds))):
        true = all_targets[i]
        pred = all_preds[i]
        file = test_filenames[i]

        print(f"File: {file}")
        print(f"  Ground Truth: x={true[0]:.3f}, y={true[1]:.3f}, w={true[2]:.3f}, h={true[3]:.3f}")
        print(f"  Prediction:   x={pred[0]:.3f}, y={pred[1]:.3f}, w={pred[2]:.3f}, h={pred[3]:.3f}\n")


# --- MAIN SCRIPT ---
def main():
    coords_df = pd.read_excel(COORDS_FILE)
    unique_files = coords_df['File_Name'].unique()

    # Split filenames
    train_files, temp_files = train_test_split(unique_files, test_size=0.4, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

    train_dataset = SignalDataset(train_files, coords_df)
    val_dataset = SignalDataset(val_files, coords_df)
    test_dataset = SignalDataset(test_files, coords_df)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=1)

    model = DefectCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print("\n--- Training CNN Model ---")
    train_model(model, train_loader, val_loader, optimizer, criterion)

    # Load best model
    model.load_state_dict(torch.load("best_defect_cnn.pt"))

    print("\n--- Evaluating on Test Set ---")
    evaluate_and_plot(model, test_loader, test_files, coords_df)


if __name__ == "__main__":
    main()
