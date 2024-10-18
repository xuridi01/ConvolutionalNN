import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import CNN

transform = transforms.Compose([transforms.ToTensor(),])
batch_size = 32

def load_mnist():
    train_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_data = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, len(test_data)

def load_cifar():
    train_data = datasets.CIFAR10(root='./data', train=True,transform=transform, download=True)
    test_data = datasets.CIFAR10(root='./data', train=False,transform=transform, download=True)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, len(test_data)

#params for nn
epochs = 10
learning_rate = 0.005

activation_function_list = [nn.functional.relu, nn.functional.sigmoid, nn.functional.elu, nn.functional.gelu, nn.functional.silu, nn.functional.selu]
activation_function_list_names = activation_fun_list_names = ['RELU', 'SIGMOID', 'ELU', 'SELU', 'GELU', 'SILU']

with open('act_fun_log.txt', 'w') as f:
    f.write(f'Experiments with different activation functions on datasets MNIST and CIFAR-10 (CNN)\n')

#MNIST
train_l, test_l, test_data_len = load_mnist()
with open('act_fun_log.txt', 'a') as f:
    f.write(f'---MNIST---\n\n')
    f.write(f'Starting params:\n-Epochs: {epochs}\n-Learning rate: {learning_rate}\n-Batch size: {batch_size}\n-Optimizer: ADAM\n-Loss function: CrossEntropy\n\n')


for fun, name in zip(activation_function_list, activation_function_list_names):
    with open('act_fun_log.txt', 'a') as f:
        f.write(f'\n{name}:\n\n')
    cnn = CNN.CNN(fun, 1, 49, 128)
    cnn.train_network(train_l, epochs, learning_rate)
    cnn.evaluate(test_l, test_data_len)

#CIFAR
train_l, test_l, test_data_len = load_cifar()
with open('act_fun_log.txt', 'a') as f:
    f.write(f'\n\n---CIFAR-10---\n\n')
    f.write(f'Starting params:\n-Epochs: {epochs}\n-Learning rate: {learning_rate}\n-Batch size: {batch_size}\n-Optimizer: ADAM\n-Loss function: CrossEntropy\n\n')

for fun, name in zip(activation_function_list, activation_function_list_names):
    with open('act_fun_log.txt', 'a') as f:
        f.write(f'\n{name}:\n\n')
    cnn = CNN.CNN(fun, 3, 64, 512)
    cnn.train_network(train_l, epochs, learning_rate)
    cnn.evaluate(test_l, test_data_len)
