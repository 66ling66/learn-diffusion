from tqdm import tqdm
import torch
from torch import nn


embedding=nn.Embedding(5,64)

num_class=torch.randn(5).long()
print(type(num_class))
print(num_class.shape)
print(embedding(num_class).shape)