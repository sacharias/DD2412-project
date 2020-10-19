from argparse import ArgumentParser
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from fixmatch.models import WRN
from fixmatch.ema import EMA

parser = ArgumentParser(description='Test a model on the test set.')
parser.add_argument('path', type=str, help='The path to the model.')
args = parser.parse_args()

testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616))
        ]
    )
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = WRN(num_classes=10).to(device)

loaded = torch.load(args.path, map_location=device)
if 'model_state_dict' in loaded:
    state_dict = loaded['model_state_dict']
else:
    state_dict = loaded
model.load_state_dict(state_dict)

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

print(f'Accuracy of the network on the {len(testset)} test images: {100 * correct / total:.2f}%')
