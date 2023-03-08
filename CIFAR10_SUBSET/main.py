import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from classifier import CIFAR10Classifier
from data import *
from train import *
import pathlib
from torch.utils.data import DataLoader
import os

data_dir = pathlib.Path("data")
target_classes = ['cat', 'deer', 'dog', 'horse']
BATCH = 128
num_cpu = os.cpu_count()
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
torch.cuda.manual_seed(42)
EPOCHS = 25

train_data = get_train_dataset(data_dir)
test_data = get_test_dataset(data_dir)

train_dataset = get_selective_subset(train_data, target_classes, 0.1)
test_dataset = get_selective_subset(test_data, target_classes, 0.1)

map_target = {subset_y: new_y for new_y, subset_y in enumerate(np.unique(train_dataset.targets))}
train_dataset.targets = map_dataset_target(train_dataset)
test_dataset.targets = map_dataset_target(test_dataset)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=num_cpu)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH, shuffle=True, num_workers=num_cpu)

model = CIFAR10Classifier(input_shape=3, output_shape=len(target_classes), hidden_units=10).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

train_func(model, train_dataloader, loss_fn, optimizer, EPOCHS, device)