import torch
import torchvision

def get_train_dataset():
    train_dataset = torchvision.datasets.CIFAR10(root="data", train=True, 
                                             download=True, transform=transform)
    return train_dataset

def get_test_dataset():
    test_dataset = torchvision.datasets.CIFAR10(root="data", train=False,
                                           download=True, transform=transform)
    return test_dataset

def get_train_dataloader():
    train_dataset = get_train_dataset()
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                            shuffle=True, num_workers=num_cpu)
    return train_dataloader    

def get_test_dataloader():     
    test_dataset = get_test_dataset()                                   
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_cpu)
    return test_dataloader

def get_class_names(dataset):
    return dataset.classes