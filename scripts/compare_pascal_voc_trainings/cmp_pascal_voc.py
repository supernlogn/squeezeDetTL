import os
import sys
import json
import numpy as np
from matplotlib import pyplot as plt
import shutil
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "src"))

from auto_ITL import start

def train_pascal(json_file):
  args = {}
  dir_path = os.path.dirname(os.path.abspath(__file__))
  args["ext_config_file_path"] = os.path.join(dir_path, json_file)
  with open(args["ext_config_file_path"], "r") as f:
    dir_name = json.load(f)["BASE_DIR"]
  
  if(os.path.exists(dir_name)):
    shutil.rmtree(dir_name)
  os.makedirs(dir_name)
  start(args)
  return dir_name

PLOT_WITH_TRAIN = True

MAP_HIST_FPATH1 = "Pascal_train_cmp1/evals/mAP_history.json"
MAP_HIST_FPATH2 = "Pascal_train_cmp2/evals/mAP_history.json"
# These results are from tensorboard
TRAIN_LOSS1_JSON = "Pascal_train_cmp1/train/total_loss.json"
TRAIN_LOSS2_JSON = "Pascal_train_cmp2/train/total_loss.json"

SAVE_FIG_TO_PATH = "/media/terabyte/projects/Thesis/transfer_learning/Pascal2TrainsCmp.pdf"

if __name__ == "__main__":
  # get train data
  # d1 = train_pascal("train_on_pascal_voc1.json")
  # d2 = train_pascal("train_on_pascal_voc2.json")

  if not PLOT_WITH_TRAIN:
    exit() 
  # read this data
  with open(MAP_HIST_FPATH1, "r") as f1:
    mAPs1 = np.array([r["mAP"] for r in json.load(f1)["history"]])
  with open(MAP_HIST_FPATH2, "r") as f2:
    mAPs2 = np.array([r["mAP"] for r in json.load(f2)["history"]])
  with open(TRAIN_LOSS1_JSON, "r") as f1:
    J = json.load(f1)
    total_loss1 = np.array([r[2] for r in J])
    total_loss_steps1 = np.array([r[1] for r in J])
  with open(TRAIN_LOSS2_JSON, "r") as f2:
    J = json.load(f2)
    total_loss2 = np.array([r[2] for r in J])
    total_loss_steps2 = np.array([r[1] for r in J])

  # To be sure that our data are aligned
  total_loss1 = total_loss1[:len(total_loss2)]
  total_loss_steps1 = total_loss_steps1[:len(total_loss2)]

  # plot this data
  fig = plt.figure()
  # total loss 1
  ax = plt.subplot(223)
  ax.plot(total_loss_steps1, total_loss1, color="xkcd:orange", linewidth=2)
  ax.set_xticks(np.arange(2000, 55002, 20000))
  ax.set_xlabel("steps")
  ax.set_ylabel("Total loss", fontsize=20)
  ax.grid()
  # total loss 2
  ax = plt.subplot(224)
  ax.plot(total_loss_steps2, total_loss2, color="xkcd:orange", linewidth=2)
  ax.set_xticks(np.arange(2000, 55002, 20000))
  ax.set_xlabel("steps")
  ax.set_ylabel("Total loss", fontsize=20)
  ax.grid()
  # mAP 1
  ax = plt.subplot(221)
  ax.plot(np.arange(6001, 60001, 2000), mAPs1, linewidth=2)
  ax.set_xticks(np.arange(6000, 60001, 20000))
  ax.set_yticks(np.arange(0, 0.5, 0.1))
  ax.set_xlabel("steps")
  ax.set_ylabel("mAP", fontsize=20)
  ax.grid()
  # mAP 2
  ax = plt.subplot(222)
  ax.plot(np.arange(6001, 60001, 2000), mAPs2, linewidth=2)
  ax.set_xticks(np.arange(6000, 60001, 20000))
  ax.set_yticks(np.arange(0, 0.5, 0.1))
  ax.set_xlabel("steps")
  ax.set_ylabel("mAP", fontsize=20)
  ax.grid()

  plt.tight_layout()
  plt.savefig(SAVE_FIG_TO_PATH)
  plt.show()
