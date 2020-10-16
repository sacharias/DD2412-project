from argparse import ArgumentParser
import matplotlib.pyplot as plt

parser = ArgumentParser(description='Plot a csv log.')
parser.add_argument('path', type=str, help='The path to the csv file.')
parser.add_argument('x', type=str, help='The attribute to use as x.')
parser.add_argument('y', type=str, help='The attribute to use as y.')
args = parser.parse_args()

with open(args.path, 'r') as f:
    lines = f.read().splitlines()

d = dict()
lines = [line.split(',') for line in lines]
for entry in zip(*lines):
    d[entry[0]] = [eval(e) for e in entry[1:]]

def plot(x, y, xlabel, ylabel):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

plot(d[args.x], d[args.y], args.x, args.y)
