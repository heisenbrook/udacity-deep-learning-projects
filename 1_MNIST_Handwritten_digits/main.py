from utils.show5 import show5
from utils.model import Model
from utils.train_test import training, testing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Define transforms
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=0.5, std=0.5)])

# Create training set and define training dataloader
training_set = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(training_set, batch_size=250, shuffle=True)

# Create test set and define test dataloader
test_set = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=250)

# Explore data
dataiter = iter(train_loader)
batch = next(dataiter)
image = batch[0][0]
print(image.numpy().shape) #added some line of code to get the shape of the images for the input layer

show5(train_loader)

model = Model()

#loss function
loss_fn = nn.CrossEntropyLoss() #loss function same as Softmax for classification purposes

#optimizer
optim_fn = optim.Adam(model.parameters())

train_loss_h = training(model, train_loader, optim_fn, loss_fn)

#plot training loss
plt.plot(train_loss_h, label="Training Loss")
plt.legend()
plt.show()

testing(model, test_loader, loss_fn)

#testing model on random samples from test set

dataiter = iter(test_loader)
batch = next(dataiter)
image = batch[0][np.random.choice(np.arange(30))]
pred_ar = model(image)
pred = torch.argmax(pred_ar)
print(int(pred))

plt.imshow(image.numpy().T.squeeze().T)
plt.show()

#save model for future loading
torch.save(model.state_dict(), 'saved_model')