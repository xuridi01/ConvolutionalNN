import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.module import T


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        #pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        #dropout
        self.dropout = nn.Dropout2d(p=0.5)

        #fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        #two conv layers with polling -> output is 64 feature maps, size 7x7
        x = self.pool(self.dropout(nn.functional.relu(self.conv1(x))))
        x = self.pool(nn.functional.relu(self.conv2(x)))

        #flattening of images x.size->batch size, -1 -> 64 * 7 * 7
        x = x.view(x.size(0), -1)

        #fully connected layers
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def train_network(self, train_loader, epochs, learning_rate):
        self.train()

        optimizer = optim.SGD(self.parameters(), learning_rate)
        # optimizer = optim.Adagrad(self.parameters(), learning_rate)
        # optimizer = optim.Adam(self.parameters(), learning_rate)

        for epoch in range(epochs):
            loss_during_train = 0

            for image, label in train_loader:
                #forward prop
                output = self.forward(image)
                loss = nn.functional.cross_entropy(output, label)

                #back prop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #logging
                loss_during_train += loss.item()

            #print logs about learning
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss_during_train / len(train_loader):.4f}')

    def evaluate(self, test_loader, ):
        self.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for image, label in test_loader:
                output = self.forward(image)

                #taking index with the highest value
                _, prediction = torch.max(output, 1)

                for i in range(len(prediction)):
                    if prediction[i] == label[i]:
                        correct += 1
                total += label.size(0)

        print(f'Accuracy: {100 * correct / total:.2f}%')