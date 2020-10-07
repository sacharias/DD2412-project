import torchvision
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch


#Placeholders for RandAugment or CTAugment
weekly_augment = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

strongly_augment = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



class unLabeled(Dataset):
    def __init__(self, data, weeklyAug, stronglyAug):
        self.data = data
        self.weeklyAug = weeklyAug
        self.stronglyAug = stronglyAug


    def __getitem__(self, index):
        x, y = self.data[index]

        #For some reason it converts the image to a tensor, so cant apply augmentations, so have to convert it back to an PIL.
        img = transforms.ToPILImage()(x)
        xW, xS = self.weeklyAug(img), self.stronglyAug(img)
        return xW, xS, y

    def __len__(self):
        return len(self.data)


def createDatasetSplit(dataset, labeledSize, validationSize, seed = 1):
    totalSize = labeledSize + validationSize

    part1, part2, part3 = torch.utils.data.random_split(dataset, [labeledSize, len(dataset) - totalSize, validationSize], generator=torch.Generator().manual_seed(seed))

    part1.dataset.transform = weekly_augment

    labeledDataset = part1

    """
    In the fixmatch paper they mention that they add the labeled dataset to the unlabeled dataset, however because we want measure
    the accuracy on the unlabeled sample through trainig it would bias the metric, therefore we dont add the labeled data.

    However to add the labeled data add the following line
    unlabledDataset = unLabeled(ConcatDataset([part1,part2]), weekly_augment, strongly_augment)
    """

    unlabledDataset = unLabeled(part2, weekly_augment, strongly_augment)

    part3.dataset.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    validationDataset = part3

    return labeledDataset, unlabledDataset, validationDataset
