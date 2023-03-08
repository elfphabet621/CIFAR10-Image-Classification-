from torch import nn
import torch

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device):
  model.train()
  train_loss, train_acc = 0, 0

  for X, y in dataloader:
    X, y = X.to(device), y.to(device)
    y_logits = model(X)

    loss = loss_fn(y_logits, y)
    train_loss += loss.item()
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
    train_acc += (y_pred == y).sum().item() / len(y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  return train_loss/len(dataloader), train_acc/len(dataloader)

def test_step(model: nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device):
  model.eval()
  with torch.inference_mode():
    test_loss, test_acc = 0, 0
    for X, y in dataloader:
      X, y = X.to(device), y.to(device)

      y_logits = model(X)
      test_loss += loss_fn(y_logits).item()

      y_pred = y_logits.softmax(dim=-1).argmax(dim=-1)
      test_acc += (y_pred == y).sum().item() / len(y)

    return test_loss / len(dataloader), test_acc / len(dataloader)
  
from tqdm.auto import tqdm
def train_func(model: nn.Module,
               train_dataloader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               epochs : int, device):
  for epoch in tqdm(range(epochs)):
    train_loss, train_acc = train_step(model, train_dataloader, 
                                      loss_fn, optimizer, device)
    test_loss, test_acc = train_step(model, train_dataloader, 
                                      loss_fn, optimizer, device)
    print(f'Epoch: {epoch}  | Train loss: {train_loss : .4f} | Train Acc: {train_acc:.2f} | Test loss {test_loss:.4f} | Test Acc {test_acc:.2f}')