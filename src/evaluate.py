import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def load_test_dataset(path: str):
    df = pd.read_csv(path)
    X = df.drop(columns=["Rented_Bike_Count"]).values.astype(np.float32)
    y = df["Rented_Bike_Count"].values.astype(np.float32).reshape(-1, 1)
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)
    return TensorDataset(X_tensor, y_tensor), df

def evaluate_model(model, test_path, batch_size=64, device="cpu", plot=True):
    model.eval()
    dataset, df_raw = load_test_dataset(test_path)
    loader = DataLoader(dataset, batch_size=batch_size)

    preds_list = []
    targets_list = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            preds_list.append(preds)
            targets_list.append(y_batch.numpy())

    y_true = np.vstack(targets_list)
    y_pred = np.vstack(preds_list)

    mse = mean_squared_error(y_true, y_pred)

    print("Evaluation Metrics on Test Set:")
    print(f"  MSE: {mse:.4f}")

    if plot:
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.xlabel("True Count")
        plt.ylabel("Predicted Count")
        plt.title("True vs Predicted Bike Rentals")
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.show()

    return y_true, y_pred