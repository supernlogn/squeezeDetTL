import os
import sys
import json

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "src"))

import numpy as np
from auto_ITL import start
from matplotlib import pyplot as plt


SAVE_FIG_TO_PATH = os.path.join(os.path.dirname(__file__), "kitti_compare.pdf")

if __name__ == "__main__":
  args = {}
  dir_path = os.path.dirname(os.path.realpath(__file__))
  args["ext_config_file_path"] = os.path.join(dir_path, "kitti_squeezeDet_config.json")
  base_dir = json.load(open(args["ext_config_file_path"]))["BASE_DIR"]
  if not os.path.exists(base_dir):
    os.makedirs(base_dir)
  start(args)

  # get training directory
  train_dir = os.path.join(base_dir, "train")

  # plot results
  with open(os.path.join(train_dir, "step_times.json"), "r") as f:
    J = json.load(f)
    steps = np.arange(J["MAX_STEPS"])
    times = np.array(J["TIMES"])
  
  fig = plt.figure()
  plt.plot(steps, times, color="xkcd:green", linewidth=2)

  """
    For comparison export sec/batch of the [old implementation](https://github.com/BichenWuUCB/squeezeDet/blob/master/src/train.py)
    and plot it for the same number of steps with the same way
    with color "xkcd:purple".
  """

  plt.grid()
  plt.set_xticks(np.arange(0, J["MAX_STEPS"], 5000))
  plt.set_xlabel("steps")
  plt.set_ylabel("total_time")

  # save figure and plot it
  plt.savefig(SAVE_FIG_TO_PATH)
  plt.show()