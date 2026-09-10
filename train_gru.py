import torch
import torch.nn as nn

from create_sequences import train_loader, val_loader, test_loader


# =========================================================
# 1. 建立 GRU 模型
# =========================================================

class VisibilityGRU(nn.Module):
    def __init__(
        self,
        input_size=16,
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

model = VisibilityGRU().to(device)

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