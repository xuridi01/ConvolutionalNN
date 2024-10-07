import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import CNN
transform = transforms.Compose([transforms.ToTensor(),])

train_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_data = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=True)

#params for nn
epochs = 10
learning_rate = 0.01


#creating, training and evaluating on test data
cnn = CNN.CNN()
cnn.train_network(train_loader, epochs, learning_rate)
cnn.evaluate(test_loader)
