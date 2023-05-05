import torch

x = torch.randn(3, 4)
print(x)

indices = torch.tensor([0, 1, 2])
print(indices)

new = torch.index_select(x, 0, indices)
print(new)

