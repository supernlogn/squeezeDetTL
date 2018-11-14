import os
import sys
from multiprocessing import Process
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "src"))

cwd = os.getcwd()

from supervisor import supervisor


if __name__ == "__main__":
  args = {}
  dir_path = os.path.dirname(os.path.abspath(__file__))
  args["ext_config_file_path"] = os.path.join(dir_path, "coco_squeezeDet_config.json")
  supervisor.start(args)