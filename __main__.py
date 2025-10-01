import torch
import torch.nn as nn
import pandas as pd
import joblib

# 1. Recreate model architecture
model = nn.Sequential(
    nn.Linear(38, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

# 2. Load trained weights
model.load_state_dict(torch.load("model.pth"))
model.eval()

# 3. Load preprocessing
scaler = joblib.load("scaler.pkl")
norm = torch.load("norm.pth")
X_mean, X_std = norm["mean"], norm["std"]

# 4. Load new data
df_new = pd.read_csv("eval.csv")
df_new = df_new.dropna()
df_new = df_new.apply(pd.to_numeric, errors="coerce").astype("float64")

df_new_scaled = pd.DataFrame(scaler.transform(df_new), columns=df_new.columns)
X_new = df_new_scaled.drop("koi_score", axis=1).values
X_new = torch.tensor(X_new, dtype=torch.float32)

# Standardize with training mean/std
#X_new = (X_new - X_mean) / X_std

# 5. Predict
with torch.no_grad():
    logits = model(X_new)
    probs = torch.sigmoid(logits)   # convert logits → probabilities
    labels = (probs > 0.5).int()

print("Probabilities:", probs.numpy().flatten())
print("Predicted labels:", labels.numpy().flatten())

