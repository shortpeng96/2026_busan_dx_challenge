import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

proj_dir = "C:\\Sandbox\\2026_busan_dx_challenge\\Ilgwang_WaterQuality_Project"
df_master = pd.read_csv(os.path.join(proj_dir, "Data_Processed", "master_dataset_ilgwang_v2.csv"))
df_weather = pd.read_csv(os.path.join(proj_dir, "Data_Raw", "ilgwang_weather_2014_2026.csv"))

df_master['date'] = pd.to_datetime(df_master['date'])
df_weather['date'] = pd.to_datetime(df_weather['date'])
df_weather = df_weather.sort_values('date').set_index('date')

# Select base weather features
weather_features = ['precip_daily', 'temp_daily', 'wind_max', 'wind_sin', 'wind_cos', 'tide_range', 'gijang_discharge_m3_day']

# Impute missing in weather using forward/backward fill
df_weather = df_weather[weather_features].ffill().bfill()

seq_length = 7
X_sequences = []
y_labels = []

# Build 3D Tensor
for idx, row in df_master.iterrows():
    end_date = row['date']
    start_date = end_date - pd.Timedelta(days=seq_length-1)
    
    # Get sequence
    seq = df_weather.loc[start_date:end_date]
    if len(seq) == seq_length:
        X_sequences.append(seq.values)
        y_labels.append(row['any_exceed'])

X = np.array(X_sequences)
y = np.array(y_labels).astype(int)

print(f"Built 3D Tensor: X={X.shape}, y={y.shape}")

# Flatten for scaling, then reshape back
n_samples, n_steps, n_features = X.shape
scaler = StandardScaler()
X_flat = scaler.fit_transform(X.reshape(-1, n_features))
X = X_flat.reshape(n_samples, n_steps, n_features)

class WaterLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=16, num_layers=1):
        super(WaterLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1 if num_layers>1 else 0)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] # Take last timestep
        out = self.fc(out)
        return self.sigmoid(out).squeeze()

# 5-Fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
aucs = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training LSTM on {device}...")

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    X_train, X_test = torch.tensor(X[train_idx], dtype=torch.float32).to(device), torch.tensor(X[test_idx], dtype=torch.float32).to(device)
    y_train, y_test = torch.tensor(y[train_idx], dtype=torch.float32).to(device), torch.tensor(y[test_idx], dtype=torch.float32).to(device)
    
    # Scale pos weight
    pos_weight = (len(y_train) - sum(y_train)) / max(1, sum(y_train))
    criterion = nn.BCELoss(weight=torch.where(y_train==1, pos_weight, 1.0))
    
    model = WaterLSTM(input_size=n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    epochs = 150
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        preds = model(X_test).cpu().numpy()
        try: auc = roc_auc_score(y_test.cpu().numpy(), preds)
        except: auc = 0
        aucs.append(auc)

final_auc = np.mean(aucs)
print(f"\\n=== FINAL RESULT ===")
print(f"LSTM Sequence Model ROC-AUC: {final_auc:.5f}")
