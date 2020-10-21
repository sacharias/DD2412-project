from argparse import ArgumentParser
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from fixmatch.models import WRN
from fixmatch.ema import EMA

def load_model(model, path):
    """Loads the model state dict."""
    loaded = torch.load(path, map_location=device)
    model.load_state_dict(loaded['model_state_dict'] if 'model_state_dict' in loaded else loaded)

def get_predictions(model):
    """Gets the predictions of a model."""
    model.eval()
    with torch.no_grad():
        predictions = torch.zeros(len(testset), dtype=torch.int64)
        offset = 0
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            _, predicted = torch.max(logits, 1)
            predictions[offset:offset+len(y)] = predicted
            offset += len(y)
        return predictions

if __name__ == '__main__':
    # Parse arguments
    parser = ArgumentParser(description='Test an ensemble on the test set.')
    parser.add_argument('dataset', type=str, help='The dataset (CIFAR-10/CIFAR-100).')
    parser.add_argument('paths', type=str, help='The paths to the models.', nargs='+')
    parser.add_argument('--batch_size', type=int, help='The batch size to use.', default=64)
    args = parser.parse_args()

    # Set dataset
    if args.dataset == 'CIFAR-10':
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
    elif args.dataset == 'CIFAR-100':
        testset = torchvision.datasets.CIFAR100(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761))
                ]
            )
        )
    else:
        raise Exception(f'Dataset "{args.dataset}" is not implemented.')

    # Set up
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = WRN(num_classes=len(testset.classes)).to(device)

    # Aggregate predictions
    one_hot_sum = torch.zeros((len(testset), len(testset.classes)), dtype=torch.int64)
    for path in tqdm(args.paths):
        load_model(model=model, path=path)
        predictions = get_predictions(model=model)
        one_hot = torch.nn.functional.one_hot(predictions, num_classes=len(testset.classes))
        one_hot_sum += one_hot

    # Calculate accuracy
    _, ensemble = torch.max(one_hot_sum, dim=1)
    targets = torch.tensor(testset.targets, dtype=torch.int64)
    correct = (ensemble == targets).sum().item()
    print(f'Correct: {correct}/{len(targets)}, Accuracy: {correct / len(targets)}')
