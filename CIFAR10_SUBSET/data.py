from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import random

def get_train_dataset(data_dir):
    train_data = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=ToTensor())
    return train_data

def get_test_dataset(data_dir):
    test_data = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=ToTensor())
    return test_data
  
def get_selective_subset(data, target_classes, num_to_get):
  class_indexes = data.class_to_idx
  targets = [class_indexes[target] for target in target_classes] 

  indices = []
  for idx, t in enumerate(data.targets):
    if t in targets:
      indices.append(idx)
  
  random.shuffle(indices)
  indices = indices[:int(num_to_get * len(indices))]
  labels = [data.targets[i] for i in indices]
  
  return custom_subset(data, indices, labels, target_classes)

def map_dataset_target(dataset, map_target):
   return list(map(lambda x: map_target[x], dataset.targets))