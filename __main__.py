print("loading libraries...")
import torch
import torch.nn as nn
import pandas as pd
import joblib


model = nn.Sequential(
    nn.Linear(34, 64),   # input → hidden
    nn.ReLU(),
    nn.Linear(64, 64),   # hidden → hidden
    nn.ReLU(),
    nn.Linear(64, 64),   # hidden → hidden
    nn.ReLU(),
    nn.Linear(64, 32),   # hidden → hidden
    nn.ReLU(),
    nn.Linear(32, 1)     # hidden → output (no Sigmoid here!)
)

print("loading model...")
model.load_state_dict(torch.load("model.pth"))
model.eval()

scaler = joblib.load("scaler.pkl")

df_new = pd.read_csv("eval.csv")
df_new = df_new.dropna()
df_new = df_new.apply(pd.to_numeric, errors="coerce").astype("float64")

df_new_scaled = pd.DataFrame(scaler.transform(df_new), columns=df_new.columns)
X_new = df_new_scaled.drop(['koi_score', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec'], axis=1).values
X_new = torch.tensor(X_new, dtype=torch.float32)

# Standardize with training mean/std
#X_new = (X_new - X_mean) / X_std

print("evaluating...")
with torch.no_grad():
    logits = model(X_new)
    probs = torch.sigmoid(logits)   # convert logits → probabilities
    labels = (probs > 0.5).int()

print("Probabilities:", probs.numpy().flatten())
print("Predicted labels:", labels.numpy().flatten())

