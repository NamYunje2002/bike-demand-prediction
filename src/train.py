# 📄 파일: src/train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib  # 스케일러 저장용

def load_dataset(path: str, scaler=None, fit=False):
    df = pd.read_csv(path)

    X = df.drop(columns=["Rented_Bike_Count"]).values.astype(np.float32)
    y = df["Rented_Bike_Count"].values.astype(np.float32).reshape(-1, 1)

    if fit:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)

    return TensorDataset(X_tensor, y_tensor), scaler

def train_model(train_path, val_path, input_dim, epochs=100, batch_size=64, lr=1e-3, device="cpu", patience=10):
    from model import BikeDemandMLP

    # 1. 학습용 데이터 로드 및 스케일러 학습
    train_data, scaler = load_dataset(train_path, scaler=None, fit=True)

    # 2. 검증용 데이터는 같은 스케일러로 transform만 수행
    val_data, _ = load_dataset(val_path, scaler=scaler, fit=False)

    # 3. DataLoader 구성
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    # 4. 모델 및 손실 함수 설정
    model = BikeDemandMLP(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    best_model_state = None
    trigger_times = 0

    # 5. 학습 루프
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                preds = model(X_val)
                loss = loss_fn(preds, y_val)
                val_loss += loss.item()

        print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # 6. 최고 모델 저장
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    torch.save(model.state_dict(), "../models/mlp_bike_demand.pt")
    joblib.dump(scaler, "../models/input_scaler.pkl")  # 정규화 스케일러도 저장

    return model
