import torch
import torch.nn as nn

x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[3.0], [5.0], [7.0], [9.0]])

model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(5000):
    optimizer.zero_grad()
    pred = model(x)

    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(epoch, loss.item())
    
print("weight = ", model.weight.item())
print("bias = ", model.bias.item())
