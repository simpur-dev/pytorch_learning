import torch
import torch.nn as nn
import math

torch.manual_seed(42)

x = torch.linspace(0, 2 * math.pi, 40).unsqueeze(1)
y = torch.sin(x) + 0.2 * torch.randn(x.size())

class BigNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

model = BigNet()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, weight_decay = 0.001)

best_loss = float('inf')
patience = 500
wait = 0

model.train()

for epoch in range(5000):
    optimizer.zero_grad()
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()

    if loss.item() < best_loss:
        best_loss = loss.item()
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f"Early Stop! 在第 {epoch} 轮停止，best_loss = {best_loss:.6f}")
            break
 
    if epoch % 500 == 0:
        print(f"epoch {epoch}, loss = {loss.item():.6f}")
 
model.eval()
 
with torch.no_grad():
    pred = model(x)
 
print("\n前 5 个预测值 vs 真实值：")
for i in range(5):
    print(f"  x={x[i].item():.2f}  预测={pred[i].item():.4f}  真实={y[i].item():.4f}")
