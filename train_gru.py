import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from create_sequences import (
    train_loader,
    val_loader,
    test_loader,
    persistence_test
)

# =========================================================
# 1. 建立 GRU 模型
# =========================================================

class VisibilityGRU(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=1
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, hidden = self.gru(x)

        last_hidden = hidden[-1]

        prediction = self.fc(last_hidden)

        return prediction


# =========================================================
# 2. 選擇 CPU / GPU
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

input_size = train_loader.dataset.tensors[0].shape[2]
model = VisibilityGRU(input_size=input_size).to(device)

print(model)


# =========================================================
# 3. 測試 DataLoader → GRU → Prediction
# =========================================================

first_X, first_y = next(iter(train_loader))

first_X = first_X.to(device)
first_y = first_y.to(device)

with torch.no_grad():
    prediction = model(first_X)

print("\n=== Model Forward Test ===")
print("Input shape:", first_X.shape)
print("Prediction shape:", prediction.shape)
print("Target shape:", first_y.shape)


# =========================================================
# 4. Loss function 與 Optimizer
# =========================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
# =========================================================
# 5. 訓練模型
# =========================================================

EPOCHS = 30

train_losses = []
val_losses = []

best_val_loss = float("inf")
best_epoch = 0

BEST_MODEL_PATH = "best_gru_vis_model.pt"


for epoch in range(EPOCHS):

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------
    model.train()

    train_loss_sum = 0.0
    train_sample_count = 0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # 清除上一個 batch 的梯度
        optimizer.zero_grad()

        # Forward
        predictions = model(X_batch)

        # 計算 Loss
        loss = criterion(
            predictions,
            y_batch
        )

        # Backpropagation
        loss.backward()

        # 更新模型參數
        optimizer.step()

        batch_size = X_batch.size(0)

        train_loss_sum += (
            loss.item() * batch_size
        )

        train_sample_count += batch_size


    train_loss = (
        train_loss_sum
        / train_sample_count
    )


    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------
    model.eval()

    val_loss_sum = 0.0
    val_sample_count = 0

    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(X_batch)

            loss = criterion(
                predictions,
                y_batch
            )

            batch_size = X_batch.size(0)

            val_loss_sum += (
                loss.item() * batch_size
            )

            val_sample_count += batch_size


    val_loss = (
        val_loss_sum
        / val_sample_count
    )


    # -----------------------------------------------------
    # 紀錄每個 Epoch 的 Loss
    # -----------------------------------------------------

    train_losses.append(train_loss)
    val_losses.append(val_loss)


    # -----------------------------------------------------
    # 儲存 Validation Loss 最佳模型
    # -----------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_epoch = epoch + 1

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )


    # -----------------------------------------------------
    # 顯示當前 Epoch
    # -----------------------------------------------------

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )


print("\n=== Training 完成 ===")

print(
    f"Best Epoch: {best_epoch}"
)

print(
    f"Best Validation Loss: "
    f"{best_val_loss:.4f}"
)


# =========================================================
# 7. 載入 Validation 表現最佳的模型
# =========================================================

model.load_state_dict(
    torch.load(
        BEST_MODEL_PATH,
        map_location=device
    )
)

model.eval()

print(
    f"\n已載入 Best Model："
    f"Epoch {best_epoch}, "
    f"Validation Loss = {best_val_loss:.4f}"
)

# =========================================================
# 8. Test Set 評估
# =========================================================

test_loss_sum = 0.0
test_sample_count = 0

all_predictions = []
all_targets = []


with torch.no_grad():

    for X_batch, y_batch in test_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        predictions = model(X_batch)

        loss = criterion(
            predictions,
            y_batch
        )

        batch_size = X_batch.size(0)

        test_loss_sum += (
            loss.item() * batch_size
        )

        test_sample_count += batch_size

        all_predictions.append(
            predictions.cpu()
        )

        all_targets.append(
            y_batch.cpu()
        )


test_loss = (
    test_loss_sum
    / test_sample_count
)


all_predictions = torch.cat(
    all_predictions
).numpy()

all_targets = torch.cat(
    all_targets
).numpy()

# =========================================================
# 9. Test RMSE / MAE
# =========================================================

rmse = np.sqrt(
    np.mean(
        (
            all_predictions
            - all_targets
        ) ** 2
    )
)

mae = np.mean(
    np.abs(
        all_predictions
        - all_targets
    )
)


print("\n=== Test Results ===")

print(
    f"Best Epoch: {best_epoch}"
)

print(
    f"Test MSE: "
    f"{test_loss:.4f}"
)

print(
    f"Test RMSE: "
    f"{rmse:.4f} km"
)

print(
    f"Test MAE: "
    f"{mae:.4f} km"
)

# =========================================================
# 10. Persistence Baseline
# =========================================================

test_targets = all_targets.squeeze()

persistence_mse = np.mean(
    (
        persistence_test
        - test_targets
    ) ** 2
)

persistence_rmse = np.sqrt(
    persistence_mse
)

persistence_mae = np.mean(
    np.abs(
        persistence_test
        - test_targets
    )
)

print("\n=== Persistence Baseline ===")

print(
    f"Persistence MSE: "
    f"{persistence_mse:.4f}"
)

print(
    f"Persistence RMSE: "
    f"{persistence_rmse:.4f} km"
)

print(
    f"Persistence MAE: "
    f"{persistence_mae:.4f} km"
)

#GRU vs Persistence

print("\n=== GRU vs Persistence ===")

print(
    f"GRU RMSE:         {rmse:.4f} km"
)

print(
    f"Persistence RMSE: {persistence_rmse:.4f} km"
)

print(
    f"GRU MAE:          {mae:.4f} km"
)

print(
    f"Persistence MAE:  {persistence_mae:.4f} km"
)

# =========================================================
# 6. 繪製 Training / Validation Loss Curve
# =========================================================

epochs = range(
    1,
    EPOCHS + 1
)

plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    train_losses,
    marker="o",
    markersize=4,
    label="Training Loss"
)

plt.plot(
    epochs,
    val_losses,
    marker="o",
    markersize=4,
    label="Validation Loss"
)

plt.axvline(
    x=best_epoch,
    linestyle="--",
    label=f"Best Epoch = {best_epoch}"
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")

plt.title(
    "GRU Training and Validation Loss"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "training_validation_loss_vis.png",
    dpi=300
)

plt.show()
