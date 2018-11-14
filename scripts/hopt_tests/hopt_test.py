import os
import sys
import json
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "src"))
from hypervisor import hyperparam_tuner
from auto_ITL import start

if __name__ == "__main__":
  dir_path = os.path.dirname(os.path.realpath(__file__))
  args = {}
  args["ext_config_file_path"] = os.path.join(dir_path, "hopt_pascal_config.json")
  start(args)