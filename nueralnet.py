import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 1) Load & engineer features
df = pd.read_csv('public_cases.csv')
X = df[['trip_duration_days','miles_traveled','total_receipts_amount']].copy()
y = df['expected_output'].values.astype(np.float32)

# optional extras (boosts convergence)
X['miles_per_day']    = X['miles_traveled'] / X['trip_duration_days']
X['receipts_per_day'] = X['total_receipts_amount'] / X['trip_duration_days']
X['is_5_day_trip']    = (X['trip_duration_days']==5).astype(int)

# 2) Pre-scale
scaler = StandardScaler()
X = scaler.fit_transform(X).astype(np.float32)

# Convert to numpy
X_np = X
y_np = y

# 3) Define the expanded PyTorch model
class WideReimbursementNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Expanded center
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Back to previous narrower layers
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.model(x)

# Training function for one fold
def train_fold(X_train, y_train, X_val, y_val, epochs=200, batch_size=32, patience=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WideReimbursementNet(X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.L1Loss()  # MAE

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    best_val_mae = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
                preds = model(xb)
                val_losses.append(criterion(preds, yb).item())
        val_mae = np.mean(val_losses)
        
        # Early stopping
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            epochs_no_improve = 0
            best_weights = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
    
    # Load best weights and return best validation MAE
    model.load_state_dict(best_weights)
    return best_val_mae

# 4) Cross-validate
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mae_scores = []

for train_idx, val_idx in kf.split(X_np):
    X_tr, X_va = X_np[train_idx], X_np[val_idx]
    y_tr, y_va = y_np[train_idx], y_np[val_idx]
    val_mae = train_fold(X_tr, y_tr, X_va, y_va)
    mae_scores.append(val_mae)

print(f"5-fold Expanded PyTorch MLP MAE: {np.mean(mae_scores):.2f} ± {np.std(mae_scores):.2f}")
