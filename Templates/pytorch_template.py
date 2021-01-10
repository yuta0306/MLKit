import torch
import torch.nn as nn
from torch.nn.modules import loss, module
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

"""
Preparing datasets
"""

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Declaring nn class function

    def forward(self, *args):
        raise NotImplementedError()

model = MyModel()
optimizer = None # optimizer
criterion = None # criterion

"""
Training
"""
EPOCH = 20
for epoch in range(EPOCH):
    # Processing datasets
    INPUT = None
    TARGET = None

    output = model(INPUT)
    loss = criterion(output, TARGET)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()   
