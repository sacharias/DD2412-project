from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from fixmatch.models import WRN

PATH = 'net-00340.pt'

testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = WRN(num_classes=10).to(device)
model.load_state_dict(torch.load(PATH, map_location=device)['model_state_dict'])

testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

correct = 0
total = 0
model.eval()
with torch.no_grad():
    for data in tqdm(testloader):
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
