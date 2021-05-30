import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def viz_paramseq(data):
    data = data.transpose()
    f, ax1 = plt.subplots(figsize=(19.2, 4.8), nrows=1)
    sns.heatmap(data, linewidths=0.05, ax=ax1, xticklabels=3, vmin=0, vmax=1, cmap='GnBu')
    plt.tick_params(labelsize=16)
    ax1.set_title('')
    ax1.set_xlabel('Sequence Index', fontdict={'family':'Times New Roman', 'size': 20})
    ax1.set_ylabel('Arguments', fontdict={'family':'Times New Roman', 'size': 20})
    ax1.set_yticklabels(
        ['type', 'color', 'rho', 'theta', 'angle', 'speed', 'burst', 'decay', 'radius',
         'strlen', 'bend', 'ways', 'span', 'snipe', 'delay'],
        rotation='horizontal', fontdict={'family':'Times New Roman', 'size': 18}
    )
    plt.show()
    plt.clf()


if __name__ == '__main__':
    viz_paramseq(np.load('../data/example.npy'))
