import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import pandas as pd

class MultiBranchCNNLSTM(nn.Module):
    def __init__(self, wifi_features, imu_features, lstm_hidden=128):
        super(MultiBranchCNNLSTM, self).__init__()
        
        # WiFi Branch (Conv1D expects [Batch, Channels, Length])
        # So input must be permuted to [Batch, Features, TimeSteps]
        self.wifi_conv = nn.Sequential(
            nn.Conv1d(in_channels=wifi_features, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # IMU Branch
        self.imu_conv = nn.Sequential(
            nn.Conv1d(in_channels=imu_features, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # After MaxPool(2) on TimeSteps=10, the new TimeSteps = 5
        # We need to reshape back to [Batch, TimeSteps, Channels] for LSTM
        # So we transpose back: [Batch, 5, 64] and [Batch, 5, 32]
        
        # Bi-LSTM
        self.lstm = nn.LSTM(input_size=64+32, hidden_size=lstm_hidden, 
                            num_layers=1, batch_first=True, bidirectional=True)
                            
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # X and Y
        )
        
    def forward(self, x_wifi, x_imu):
        # x_wifi: [B, T, F_w] -> [B, F_w, T]
        x_wifi = x_wifi.permute(0, 2, 1)
        # x_imu: [B, T, F_i] -> [B, F_i, T]
        x_imu = x_imu.permute(0, 2, 1)
        
        out_wifi = self.wifi_conv(x_wifi) # [B, 64, T/2]
        out_imu = self.imu_conv(x_imu)    # [B, 32, T/2]
        
        # [B, 64+32, T/2] -> [B, T/2, 96]
        out = torch.cat((out_wifi, out_imu), dim=1).permute(0, 2, 1)
        
        lstm_out, (hn, cn) = self.lstm(out) # lstm_out: [B, T/2, hidden*2]
        
        # Take the output of the last timestep
        last_out = lstm_out[:, -1, :] 
        
        # Dense
        predictions = self.fc(last_out)
        return predictions

def main():
    print("Loading preprocessed tensors...")
    base_dir = '../Datasets/dl_processed'
    
    try:
        X_train_wifi = np.load(os.path.join(base_dir, 'X_train_wifi.npy'))
        X_train_imu = np.load(os.path.join(base_dir, 'X_train_imu.npy'))
        y_train = np.load(os.path.join(base_dir, 'y_train.npy'))
        
        X_test_wifi = np.load(os.path.join(base_dir, 'X_test_wifi.npy'))
        X_test_imu = np.load(os.path.join(base_dir, 'X_test_imu.npy'))
        y_test = np.load(os.path.join(base_dir, 'y_test.npy'))
    except Exception as e:
        print(f"Error loading files: {e}. Please run preprocess_dl_pipeline.py first.")
        return

    # Convert to PyTorch Tensors
    X_train_wifi_t = torch.tensor(X_train_wifi, dtype=torch.float32)
    X_train_imu_t = torch.tensor(X_train_imu, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    
    X_test_wifi_t = torch.tensor(X_test_wifi, dtype=torch.float32)
    X_test_imu_t = torch.tensor(X_test_imu, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    
    # Create DataLoaders
    batch_size = 32
    train_dataset = TensorDataset(X_train_wifi_t, X_train_imu_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = TensorDataset(X_test_wifi_t, X_test_imu_t, y_test_t)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    wifi_features = X_train_wifi.shape[2]
    imu_features = X_train_imu.shape[2]
    
    model = MultiBranchCNNLSTM(wifi_features, imu_features)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 50
    best_loss = float('inf')
    patience = 5
    counter = 0
    
    print("Training started...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for b_w, b_i, b_y in train_loader:
            optimizer.zero_grad()
            preds = model(b_w, b_i)
            loss = criterion(preds, b_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * b_w.size(0)
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for b_w, b_i, b_y in test_loader:
                preds = model(b_w, b_i)
                loss = criterion(preds, b_y)
                test_loss += loss.item() * b_w.size(0)
        test_loss /= len(test_loader.dataset)
        
        print(f"Epoch {epoch+1:02d} | Train MSE: {train_loss:.4f} | Test MSE: {test_loss:.4f}")
        
        # Early Stopping
        if test_loss < best_loss:
            best_loss = test_loss
            counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break
                
    print("Training finished. Evaluating best model...")
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))
    model.eval()
    
    all_preds = []
    all_truths = []
    with torch.no_grad():
        for b_w, b_i, b_y in test_loader:
            preds = model(b_w, b_i)
            all_preds.append(preds.numpy())
            all_truths.append(b_y.numpy())
            
    all_preds = np.vstack(all_preds)
    all_truths = np.vstack(all_truths)
    
    # Calculate Euclidean Error
    errors = np.sqrt(np.sum((all_preds - all_truths)**2, axis=1))
    mean_err = np.mean(errors)
    max_err = np.max(errors)
    print(f"--- RESULTS ON TEST SET (S9+) ---")
    print(f"Mean Error: {mean_err:.2f} meters")
    print(f"Max Error: {max_err:.2f} meters")
    
    # Save predictions
    results_df = pd.DataFrame({
        'True_X': all_truths[:, 0],
        'True_Y': all_truths[:, 1],
        'Pred_X': all_preds[:, 0],
        'Pred_Y': all_preds[:, 1],
        'Error': errors
    })
    results_df.to_csv('../Datasets/Metrics_S9_DeepLearning.csv', index=False)
    
if __name__ == '__main__':
    main()
