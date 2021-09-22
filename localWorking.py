# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 14:45:09 2021

@author: steph
"""
import torch

def activation(x):
    """Sigmoid activation function 
        Arguments:
        x: torch.Tensor
    """
    return 1 / (1 + torch.exp(-x))

torch.manual_seed(7)
features = torch.randn((1,5))
weights = torch.randn_like(features)
bias = torch.randn((1, 1))


output = activation(torch.sum(features * weights) + bias)
a, b = weights.shape

output = activation(torch.mm(features, weights.view(b, a)) + bias)

torch.manual_seed(7)
features = torch.randn((1,3))

#Define the size of each layer in the network
n_input = features.shape[1]
n_hidden = 2
n_output = 1

# Weights for inputs to hidden layer
W1 = torch.randn(n_input, n_hidden)
# Weights for hidden layer to output layer
W2 = torch.randn(n_hidden, n_output)

# bias terms
B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))

h1 = activation(torch.mm(features, W1) + B1)
h2 = activation(torch.mm(h1, W2) + B2)

# Neural networks in PyTorch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5)),])

trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)

plt.imshow(images[1].numpy().squeeze(), cmap="Greys_r")

features = torch.flatten(images, start_dim=1)
inputs = images.view(images.shape[0], -1)

n_input = features.shape[1]
n_hidden = 256
n_output = 10

# Weights for inputs to hidden layer
W1 = torch.randn(n_input, n_hidden)
# Weights for hidden layer to output layer
W2 = torch.randn(n_hidden, n_output)

# bias terms
B1 = torch.randn(n_hidden)
B2 = torch.randn(n_output)

h1 = activation(torch.mm(features, W1) + B1)
h2 = torch.mm(h1, W2) + B2

def softmax(x):
    """Softmax activation function 
        Arguments:
        x: torch.Tensor
    """
    # key being able to work out how to correctly call vector operations, using dim and view can help
    return torch.exp(x) / torch.sum(torch.exp(x), dim=1).view(-1,1) 

sol = softmax(h2)
print(sol.shape)
print(sol.sum(dim=1))

from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch import optim

class Network(nn.Module):
    def __init__(self):
        super().__init__() # this is done to let PyTorch know to track the following additions to the network
        
        # Inputs to hidden layer linear transformation - this automatically allocates the weights to be assigned to the inputs to this linear transformation
        self.hidden_1 = nn.Linear(784, 128)
        # second hidden layer
        self.hidden_2 = nn.Linear(128, 64)
        # Output layer , 10 units - one for each digit
        self.output = nn.Linear(64, 10)
        
    def forward(self, x):
        # Pass the input tensor through each of our operations - feed forward
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        x = F.softmax(self.output(x), dim=1)
        
        return x

model = Network()

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5)),])

trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Build a feed-forward network
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))   # Note that we don't take the softmax function here

# Define the loss
criterion = nn.NLLLoss()

# Get our data
images, labels = next(iter(trainloader))
# Flatten images
images = images.view(images.shape[0], -1)

# Forward pass, get out logits
logits = model(images)
# Calculate the loss with the logits and the labels
loss = criterion(logits, labels)
print(loss)

model[0].weight.grad
loss.backward()

# Optimizers require the parameters to optimise and a learning rate
optimizer = optim.SGD(model.parameters(), lr=0.003)

print("Initial weights - ", model[0].weight)

images, labels = next(iter(trainloader))
images.resize_(64, 784)

# Clear gradients
optimizer.zero_grad()

# Forward pass, then backward pass, then update weights
output = model.forward(images)
loss = criterion(output, labels)
loss.backward()
print("Gradient - ", model[0].weight.grad)
# Take an update step and view new weights
optimizer.step()
print("Updated weights - ", model[0].weight)

# Train model for several epochs

model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1)) 

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)

epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        images = images.view(images.shape[0], -1)
        
        optimizer.zero_grad()
        
        output = model.forward(images)
        loss = criterion(output, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")
        
# See predictions

import helper

images, labels = next(iter(trainloader))
img = images[0].view(1, 784)
# Turn off gradients to speed up this part
with torch.no_grad():
    logits = model.forward(img)
    
# Outputs of the network are logits, need to take softmax for probabilities
ps = F.softmax(logits, dim=1)
helper.view_classify(img.view(1, 28, 28), ps)
