import os
from argparse import ArgumentParser
from datetime import datetime
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision
from torchvision import transforms

import fixmatch
from fixmatch.data_handler import create_dataset_split
from fixmatch.models import WRN
from fixmatch.ema import EMA

# Parse arguments
parser = ArgumentParser(description='Train using FixMatch.')
parser.add_argument('--dataset', type=str, help='The dataset to use.', default='CIFAR-10')
parser.add_argument('--labeled_size', type=int, help='Number of labeled samples.', default=4000)
parser.add_argument('--validation_size', type=int, help='Number of validation samples.', default=10000)
parser.add_argument('--mu', type=int, help='There is a 1:mu ratio between labeled and unlabeled samples.', default=7)
parser.add_argument('--batch_size', type=int, help='The size of a labeled batch.', default=64)
parser.add_argument('--epochs', type=int, help='The number of epochs.', default=1024)
parser.add_argument('--threshold', type=float, help='The confidence threshold for pseudo labels.', default=0.95)
parser.add_argument('--lambda_u', type=float, help='The weight for the unlabled loss.', default=1.0)
parser.add_argument('--decay', type=float, help='The EMA decay.', default=0.999)
parser.add_argument('--seed', type=int, help='The random seed.', default=1337)
parser.add_argument('--history', type=str, help='The directory where the training history should be saved.', default=f'history/train-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}')
args = parser.parse_args()

# Seed the RNG
torch.manual_seed(args.seed)

# Download dataset
if args.dataset == 'CIFAR-10':
    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=None
    )
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616))
elif args.dataset == 'CIFAR-100':
    trainset = torchvision.datasets.CIFAR100(
          root='./data',
          train=True,
          download=True,
          transform=None
      )
    normalize = transforms.Normalize((0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761))

else:
    raise Exception(f'Dataset "{args.dataset}" is not implemented.')

# Split dataset
labeled_dataset, unlabled_dataset, validation_dataset = create_dataset_split(
    dataset=trainset,
    labeled_size=args.labeled_size,
    validation_size=args.validation_size,
    normalize=normalize
)


# Create dataloaders
labeled_dataloader = DataLoader(labeled_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
unlabled_dataloader = DataLoader(unlabled_dataset, batch_size=args.batch_size*args.mu, shuffle=True, num_workers=4, pin_memory=True) if args.mu > 0 else None
validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Create history directory
os.mkdir(args.history)

# Print and save info
info = '\n'.join([
    'Starting training with:',
    f'    Size of labeled dataset: {len(labeled_dataset)}',
    f'    Size of unlabeled dataset: {len(unlabled_dataset)}',
    f'    Size of validation dataset: {len(validation_dataset)}',
    f'    Training using device: {device}',
    f'    Saving history to: {args.history}',
    f'\nArgs: {vars(args)}'
])
print(info)
with open(os.path.join(args.history, 'info.txt'), 'w') as f:
    f.write(info + '\n')

model = WRN(num_classes=len(trainset.classes)).to(device)

optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=0.0005,  nesterov=True)
ema_model = EMA(args.decay, model, device)

train_loss, pseudolabel_acc = fixmatch.train(
    net=model,
    labeled_dataloader=labeled_dataloader,
    unlabeled_dataloader=unlabled_dataloader,
    validation_dataloader=validation_dataloader,
    optimizer=optimizer,
    threshold=args.threshold,
    lambda_u=args.lambda_u,
    epochs=args.epochs,
    ema_model=ema_model,
    device=device,
    log_file=os.path.join(args.history, 'log.csv'),
    weight_dir=args.history
)
