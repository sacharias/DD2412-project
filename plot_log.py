from argparse import ArgumentParser
import matplotlib.pyplot as plt

parser = ArgumentParser(description='Plot a csv log.')
parser.add_argument('path', type=str, help='The path to the csv file.')
parser.add_argument('x', type=str, help='The attribute to use as x.')
parser.add_argument('y', type=str, help='The attribute to use as y.')
parser.add_argument('--xlabel', type=str, help='The label for the x-axis.')
parser.add_argument('--ylabel', type=str, help='The label for the y-axis.')
parser.add_argument('--title', type=str, help='The title for the figure.')
parser.add_argument('--stepsize', type=int, help='The stepsize for the x and y lists.', default=1)
parser.add_argument('--save', type=str, help='The file name to save as.')
args = parser.parse_args()

args.xlabel = args.x if args.xlabel is None else args.xlabel
args.ylabel = args.y if args.ylabel is None else args.ylabel

with open(args.path, 'r') as f:
    lines = f.read().splitlines()

d = dict()
lines = [line.split(',') for line in lines]
for entry in zip(*lines):
    d[entry[0]] = [eval(e) for e in entry[1:]]

def plot(x, y, xlabel, ylabel):
    plt.figure()
    plt.plot(x[0:-1:args.stepsize], y[0:-1:args.stepsize])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if args.title is not None:
        plt.title(args.title)
    if args.save is not None:
        plt.savefig(args.save)
    plt.show()

plot(d[args.x], d[args.y], args.xlabel, args.ylabel)
