import matplotlib.pyplot as plt

dataset = 'CIFAR-10'
labeled_samples = 250
x_key, y_key = 'step', 'val_acc'
stepsize = 126
alpha = 0.7

dataset_savename = ''.join(word.lower() for word in dataset.split('-'))
paths = [
    f'backup/{dataset_savename}-l{labeled_samples}/log.csv',
    f'backup/{dataset_savename}-l{labeled_samples}-u0/log.csv'
]
legends = [
    'With unlabeled data',
    'Without unlabeled data'
]
word_map = {
    'step': 'Step',
    'val': 'Validation',
    'train': 'Training',
    'acc': 'accuracy',
    'loss': 'loss'
}
x_label, y_label = word_map[x_key], ' '.join([word_map[word] for word in y_key.split('_')])
title = f'{y_label} on CIFAR-100 with {labeled_samples} labeled samples'
save = f'report/img/{dataset_savename}-l{labeled_samples}-with_without_unlabeled-{y_key}.svg'

files = []
for path in paths:
    with open(path, 'r') as f:
        files.append(f.read().splitlines())

ds = []
for lines in files:
    d = dict()
    lines = [line.split(',') for line in lines]
    for entry in zip(*lines):
        d[entry[0]] = [eval(e) for e in entry[1:]]
    ds.append(d)

# Plot
plt.figure()
for d, legend in zip(ds, legends):
    x, y = d[x_key][0:-1:stepsize], d[y_key][0:-1:stepsize]
    plt.plot(x, y, label=legend, alpha=0.75)
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.title(title)
plt.legend()
plt.savefig(save)
plt.show()
