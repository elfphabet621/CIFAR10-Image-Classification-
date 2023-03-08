import torch
import torchvision
from torch import nn

def train_step(model, train_dataloader, optimizer, loss):
    for i, (img, label) in enumerate(train_dataloader):
        img, label = img.to(device), label.to(device)

        y_logits = model(img)
        loss = loss_fn(y_logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 300 == 0:
            print(f'Epoch {epoch+1}/{num_epochs} | Step {i+1}/{n_total_steps}, Loss {loss.item():.4f}')

def eval_step(model, test_dataloader, loss):
  n_correct, n_samples = 0, 0
  n_class_correct = [0 for i in range(10)]
  n_class_samples = [0 for i in range(10)]

  for imgs, labels in test_dataloader:
    imgs, labels = imgs.to(device), labels.to(device)
    y_logits = model(imgs)

    preds = torch.argmax(y_logits, dim=1)
    n_samples += labels.size(0)
    n_correct += (preds == labels).sum().item()

    for i in range(len(labels)):
      label = labels[i]
      pred = preds[i]

      if (label == pred):
        n_class_correct[label] += 1
      n_class_samples[label] += 1

  acc = 100.0 * n_correct / n_samples
  print(f"Accuracy: {acc} %")

  for i in range(10):
    acc = 100.0 * n_class_correct[i] / n_class_samples[i]
    print(f"Accuracy of {class_names[i]} : {acc}%")