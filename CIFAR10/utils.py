import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
  img = img / 2 + 0.5
  npimg = img.permute(1,2,0).numpy()
  plt.figure(figsize=(10, 10))
  plt.imshow(npimg)
  plt.show
  plt.axis(False);

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
])