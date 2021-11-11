from __future__ import print_function

import os

import numpy as np
from MulticoreTSNE import MulticoreTSNE as TSNE

import matplotlib.pyplot as plt

ps_data = np.load("./features/ps.pth.npy").transpose(1, 0, 2, 3).reshape(16, -1).transpose()
fs_data = np.load("./features/fs.pth.npy").transpose(1, 0, 2, 3).reshape(16, -1).transpose()
mt_data = np.load("./features/meanteacher.pth.npy").transpose(1, 0, 2, 3).reshape(16, -1).transpose()
proposed_data = np.load("./features/udaiic.pth.npy").transpose(1, 0, 2, 3).reshape(16, -1).transpose()
target = np.load("./features/target.pth.npy").squeeze(1).reshape(-1)

input_target = target[::70]
# input_target = None

def draw_pictures(input_data, input_target, save_name, show_legend=False):
    tsne = TSNE(n_jobs=8, n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(input_data, None)
    label = ["Background", "LV", "Myo", "RV"]
    colors = ["tab:gray", 'tab:blue', 'tab:orange', 'tab:green', ]
    plt.figure(figsize=(4, 4))
    for i, (l, c) in enumerate(zip(label, colors)):
        index = input_target == i
        plt.scatter(*tsne_results[index].transpose(), c=c, label=l, linewidth=0)
    if show_legend:
        plt.legend()
    plt.savefig(os.path.join("features", save_name), bbox_inches="tight", dpi=180)
    plt.close()


input_data = ps_data[::70]
draw_pictures(input_data, input_target, "ps.png")

input_data = fs_data[::70]
draw_pictures(input_data, input_target, "fs.png")

input_data = mt_data[::70]
draw_pictures(input_data, input_target, "mt.png")

input_data = proposed_data[::70]
draw_pictures(input_data, input_target, "pp.png", show_legend=True)
