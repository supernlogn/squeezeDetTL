import os
import sys
import json
import numpy as np
from matplotlib import pyplot as plt
import shutil


PLOT_WITH_TRAIN = False

MAP_HIST_FPATH1 = "kitti_tests_new/evals/mAP_history.json"
MAP_HIST_FPATH2 = "kitti_greedy_bipartite/evals/mAP_history.json"
# These results are from tensorboard
TRAIN_LOSS1_JSON = "kitti_tests_new/train/total_loss.json"
TRAIN_LOSS2_JSON = "kitti_greedy_bipartite/train/total_loss.json"

SAVE_FIG_TO_PATH = "/media/terabyte/projects/Thesis/paper/figures/kittiCmpAnchorMatching.pdf"

if __name__ == "__main__":
  with open(MAP_HIST_FPATH1, "r") as f1:
    mAPs1 = np.array([r["mAP"] for r in json.load(f1)["history"]])
  with open(MAP_HIST_FPATH2, "r") as f2:
    mAPs2 = np.array([r["mAP"] for r in json.load(f2)["history"]])
  with open(TRAIN_LOSS1_JSON, "r") as f1:
    J = json.load(f1)
    total_loss1 = np.array([r[2] for r in J])
    total_loss_steps1 = np.array([r[1] for r in J])
    T1 = np.array([r[0] for r in J])
    T1 = T1 - T1[0]
  with open(TRAIN_LOSS2_JSON, "r") as f2:
    J = json.load(f2)
    total_loss2 = np.array([r[2] for r in J])
    total_loss_steps2 = np.array([r[1] for r in J])
    T2 = np.array([r[0] for r in J])
    T2 = T2 - T2[0]

  # plot this data
  fig = plt.figure()
  X = np.arange(6001, 100001, 25000)
  T1 = T1[1::(len(T1)/len(X))]
  T2 = T2[1::(len(T2)/len(X))]
  # total loss 1
  ax = plt.subplot(223)
  ax.plot(total_loss_steps1, total_loss1, color="xkcd:orange", linewidth=2)
  ax.set_xticks(X)
  ax.set_xticklabels([str(x/1000)+"k" for x in X])
  ax.set_xlabel("steps")
  ax.set_ylabel("Total loss", fontsize=20)
  ax.grid()
  # total loss 2
  ax = plt.subplot(224)
  ax.plot(total_loss_steps2, total_loss2, color="xkcd:orange", linewidth=2)
  ax.set_xticks(X)
  ax.set_xticklabels([str(x/1000)+"k" for x in X])
  ax.set_xlabel("steps")
  ax.set_ylabel("Total loss", fontsize=20)
  ax.grid()
  # mAP 1
  ax = plt.subplot(221)
  ax.set_ylim(0, 1)
  par2 = ax.twiny()
  # par2.set_ylim(ax.get_ylim())
  ax.set_xticks(X)
  par2.set_xticks(X)
  par2.set_xticklabels(["%.1fh"%(x/3600.0) for x in T1])
  # ax.set_xticks(X, minor=True)
  _Xticklabels = [(str(x[0]/1000)+"k", (x[1]/3600.0)) for x in zip(X, T1)]
  Xticklabels = []
  va = []
  for x in _Xticklabels:
    Xticklabels.append(x[0])
  ax.set_xticklabels(Xticklabels)
  
  # ax.set_yticklabels(np.arange(0, 0.45, 0.1))
  ax.set_xlabel("steps")
  par2.set_xlabel("time")
  ax.set_ylabel("mAP", fontsize=20)
  print(mAPs1[0])
  par2.plot(np.arange(6001, 100001, 2000), mAPs1, linewidth=2)
  ax.set_yticks(np.arange(0, 1.0, 0.2))
  # par2.set_yticks(np.arange(0.225, 0.45, 0.1))
  ax.grid()

  # mAP 2
  ax = plt.subplot(222)
  ax.set_ylim(0, 1)
  par2 = ax.twiny()

  ax.plot(np.arange(6001, 100001, 2000), mAPs2, linewidth=2)
  ax.set_xticks(X)
  par2.set_xticks(X)
  par2.set_xticklabels(["%.1fh"%(x/3600.0) for x in T2])
  ax.set_xticklabels([str(x/1000)+"k" for x in X])
  ax.set_yticks(np.arange(0, 1.0, 0.2))
  ax.set_xlabel("steps")
  par2.set_xlabel("time")
  ax.set_ylabel("mAP", fontsize=20)
  ax.grid()

  plt.tight_layout()
  plt.savefig(SAVE_FIG_TO_PATH)
  plt.show()