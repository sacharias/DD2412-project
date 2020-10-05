from dataHandler import createDatasetSplit
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import CIFAR10
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch
from tqdm import tqdm


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=None)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=None)


B = 1000
validationSize = 10
mu = 7
batchSize = 64


labeledDataset, unlabledDataset, validationDataset = createDatasetSplit(trainset,B,validationSize,10)

labeledDataloader = DataLoader(labeledDataset, batch_size=batchSize, shuffle=True, num_workers=16, pin_memory=True)
unlabledDataloader = DataLoader(unlabledDataset, batch_size=batchSize*mu, shuffle=True, num_workers=16, pin_memory=True)
validationDataLoader = DataLoader(validationDataset, batch_size=batchSize, shuffle=False, num_workers=16, pin_memory=True)

print("labeledDataset", len(labeledDataset))
print("unlabledDataset", len(unlabledDataset))
print("validationDataset", len(validationDataset))
print("testDataset", len(testset))


#First variable is the weekly augment image and the second is the strongly augmented image
for pic1,pic2,y in tqdm(unlabledDataloader):
    pass
    print("unlabeled yo")

for pic1,y in tqdm(labeledDataloader):
    pass
    print("labeled yo")
