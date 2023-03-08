import torch
import torchvision
from torch import nn
import os
import matplotlib.pyplot as plt
import numpy as np
from utils import transform
from model import ConvNet

device = "cuda" if torch.cuda.is_available() else "cpu"

num_epochs = 3
batch_size = 32
num_cpu = os.cpu_count()

train_dataloader = get_train_dataloader()
test_dataloader = get_test_dataloader()
class_names = get_class_names(train_dataset)

model = ConvNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
n_total_steps = len(train_dataloader)

## Train
model.train()
for epoch in range(num_epochs):
    train_step(model, train_dataloader, optimizer, loss)

## Evaluation
model.eval()
with torch.no_grad():
    eval_step(model, test_dataloader, loss)
    