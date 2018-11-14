# before using this file, you should have a json file for all the mAPs
# and a json file for the total loss
import numpy as np
from matplotlib import pyplot as plt
import json

# This file is the mAP history file
MAP_HIST_FPATH = "/media/terabyte/projects/Thesis/paper/figures/results_without_TL.json"
# download this json by using tensorboard
TRAIN_LOSS_JSON = "/media/terabyte/projects/Thesis/paper/figures/btrain2_total_loss.json"
# where to save the final figure. File extension is included.
SAVE_FIG_TO_PATH = "/media/terabyte/projects/Thesis/paper/figures/Btrain2.pdf"

if __name__ == "__main__":
  with open(MAP_HIST_FPATH, "r") as f1:
    mAPs = np.array([r["mAP"] for r in json.load(f1)])
  with open(TRAIN_LOSS_JSON, "r") as f2:
    J = json.load(f2)
    total_loss = np.array([r[2] for r in J])
    total_loss_steps = np.array([r[1] for r in J])
  fig = plt.figure()

  # total loss
  ax = plt.subplot(121)
  ax.plot(total_loss_steps, total_loss, color="xkcd:orange", linewidth=2)
  ax.set_xticks(np.arange(0, 55002, 15000))
  ax.set_xlabel("steps")
  ax.set_ylabel("Total loss", fontsize=20)
  ax.grid()
  # mAP
  ax = plt.subplot(122)
  ax.plot(np.arange(10001, 55002, 1000), mAPs, linewidth=2)
  ax.set_xticks(np.arange(10000, 55002, 15000))
  ax.set_yticks(np.arange(0, 0.40, 0.05))
  ax.set_xlabel("steps")
  ax.set_ylabel("mAP", fontsize=20)
  ax.grid()
  plt.tight_layout()
  plt.savefig(SAVE_FIG_TO_PATH)
  plt.show()
