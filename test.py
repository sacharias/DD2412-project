from dataHandler import createDatasetSplit
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import CIFAR10
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch
from tqdm import tqdm
from train import train
import torch.optim as optim
from simplemodel import Net


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=None)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=None)


B = 1000
validationSize = 10
mu = 7
batchSize = 64


labeledDataset, unlabledDataset, validationDataset = createDatasetSplit(trainset,B,validationSize,10)

labeledDataloader = DataLoader(labeledDataset, batch_size=batchSize, shuffle=True, num_workers=4, pin_memory=True)
unlabledDataloader = DataLoader(unlabledDataset, batch_size=batchSize*mu, shuffle=True, num_workers=4, pin_memory=True)
validationDataLoader = DataLoader(validationDataset, batch_size=batchSize, shuffle=False, num_workers=4, pin_memory=True)
testloader = DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=4, pin_memory=True)


print("labeledDataset", len(labeledDataset))
print("unlabledDataset", len(unlabledDataset))
print("validationDataset", len(validationDataset))
print("testDataset", len(testset))


model = Net()
epochs = 10
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
threshold = 0.95
lambda_u = 1

for epoch in range(1, epochs + 1):
      train_loss = train(model, labeledDataloader,unlabledDataloader, optimizer, threshold, lambda_u)




correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
