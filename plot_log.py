from argparse import ArgumentParser
import matplotlib.pyplot as plt

parser = ArgumentParser(description='Plot a csv log.')
parser.add_argument('path', type=str, help='The path to the csv file.')
args = parser.parse_args()

with open(args.path, 'r') as f:
    lines = f.read().splitlines()[1:]
lines = [line.split(',') for line in lines]
epochs, losses, pseudos = zip(*lines)
epochs = [int(epoch) for epoch in epochs]
losses = [float(loss) for loss in losses]
pseudos = [float(pseudo) for pseudo in pseudos]

print(pseudos)

def plot(x, y, xlabel, ylabel):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

plot(epochs, pseudos, 'Epoch', 'Accuracy of pseudo labels')
plot(epochs, losses, 'Epoch', 'Loss')
