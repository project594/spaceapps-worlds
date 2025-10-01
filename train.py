print("loading libraries...")
import pandas as pd
import numpy as np
import math

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import joblib

epochs = 200000

model = nn.Sequential(
    nn.Linear(34, 64),   # input → hidden
    nn.ReLU(),
    nn.Linear(64, 64),   # hidden → hidden
    nn.ReLU(),
    nn.Linear(64, 64),   # hidden → hidden
    nn.ReLU(),
    nn.Linear(64, 64),   # hidden → hidden
    nn.ReLU(),
    nn.Linear(64, 32),   # hidden → hidden
    nn.ReLU(),
    nn.Linear(32, 1)     # hidden → output (no Sigmoid here!)
)

loss_function = nn.BCEWithLogitsLoss()   # expects raw logits
optimizer = optim.SGD(model.parameters(), lr=0.004)

print("loading data...")

scaler = MinMaxScaler(feature_range=(0, 1))

df = pd.read_csv('cumulative_2025.09.29_09.21.24.csv')
df = df.dropna()
df = df.apply(pd.to_numeric, errors="coerce").astype("float64")

df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

features = df_scaled.drop(['koi_score', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec'], axis=1).values
labels = df_scaled['koi_score'].values

X = torch.tensor(features, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)

X_mean, X_std = X.mean(), X.std()


losses = []
for epoch in range(epochs):
    # Forward pass
    logits = model(X)         # raw outputs
    loss = loss_function(logits, y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if (epoch+1) % 10 == 0:
        print("", end=f"\rtraining...: {math.floor(epoch/epochs * 100)} %", "Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "model.pth")
joblib.dump(scaler, "scaler.pkl")
torch.save({"mean": X_mean, "std": X_std}, "norm.pth")

print("saved.")
