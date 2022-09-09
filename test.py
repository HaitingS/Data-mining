import torch
a=torch.tensor([[1,2],[1,2]])
b=torch.tensor([[3,4],[3,4]])
c=torch.stack([a,b])
print(c)
print(c.size())
print(c[2])
