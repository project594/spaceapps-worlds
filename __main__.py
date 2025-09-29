print("loading libraries...")
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

model = nn.Sequential(nn.Linear(in_features=34, out_features=1), nn.Sigmoid())
scaler = MinMaxScaler(feature_range=(0, 1))


epochs = 2000000


def predict(csv_file, model, scaler, X_train_mean, X_train_std, has_labels=True):
    # Load CSV
    df_new = pd.read_csv(csv_file)
    df_new = df_new.dropna()
    df_new = df_new.apply(pd.to_numeric, errors="coerce").astype("float64")

    # Scale with the SAME scaler
    df_new_scaled = pd.DataFrame(scaler.transform(df_new), columns=df_new.columns)

    # Extract features
    if has_labels:
        X_new = df_new_scaled.drop(['koi_score', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec'], axis=1).values
    else:
        X_new = df_new_scaled.values

    # Convert to tensor
    X_new = torch.tensor(X_new, dtype=torch.float32)

    # Standardize like training data
    #X_new = (X_new - X_train_mean) / X_train_std

    # Predict
    with torch.no_grad():
        probs = model(X_new).numpy().flatten()
        labels = (probs > 0.65).astype(int)

    return probs, labels


print("loading data...")
df = pd.read_csv('cumulative_2025.09.29_09.21.24.csv')
df = df.dropna()
df = df.apply(pd.to_numeric, errors="coerce").astype("float64")
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
print(df_scaled)


features = df_scaled.drop(['koi_score', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec'], axis=1).values  # X
labels = df_scaled['koi_score'].values # Y


X = torch.tensor(features, dtype = torch.float32)
y = torch.tensor(labels, dtype = torch.float32)
y = torch.FloatTensor(y).unsqueeze(-1)

#X = (X - X.mean())/X.std()


loss_function = nn.BCELoss() #binary classification loss
optimizer = optim.SGD(model.parameters(), lr = 0.1)


losses = []
for epoch in range(epochs):
  yhat = model(X)
  loss = loss_function(yhat, y)
  optimizer.zero_grad()
  loss.backward()
  losses.append(loss.item())
  optimizer.step()
  print("", end=f"\rtraining...: {math.floor(epoch/epochs * 100)} %")


print("\nevaluating...")
probs, labels = predict("eval.csv", model, scaler, X.mean(), X.std())
print("Probabilities:", probs)
print("Predicted labels:", labels)
