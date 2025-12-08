import torch
import torch.nn as nn

# Test loss computation
model = nn.Linear(10, 2)
criterion = nn.CrossEntropyLoss()

x = torch.randn(32, 10)
y = torch.randint(0, 2, (32,))

output = model(x)
loss = criterion(output, y)

print(f"Output shape: {output.shape}")
print(f"Loss shape: {loss.shape}")
print(f"Loss value: {loss.item()}")
print(f"Loss is scalar: {loss.dim() == 0}")
