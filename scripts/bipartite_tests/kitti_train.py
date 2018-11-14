import os
import sys
import json
from multiprocessing import Process

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "src"))

from auto_ITL import start

def launchTensorBoard(tensorBoardPath):
  import os
  os.system('tensorboard --logdir ' + tensorBoardPath)
  return

if __name__ == "__main__":
  args = {}
  dir_path = os.path.dirname(os.path.realpath(__file__))
  args["ext_config_file_path"] = os.path.join(dir_path, "kitti_squeezeDet_config.json")
  if not os.path.exists(json.load(open(args["ext_config_file_path"]))["BASE_DIR"]):
    os.makedirs(json.load(open(args["ext_config_file_path"]))["BASE_DIR"])
  start(args)
  # launch training
  # p1 = Process(target = supervisor.trainer.start, args = (args,))
  # p1.start()
  # launch evaluation
  # p2 = Process(target = supervisor.evaluator.start, args = (args,))
  # p2.start()
  # launch tensorboard
  # p3 = Process(target = launchTensorBoard, args = (args["TRAIN_DIR"],))
  # p3.start()

  # p1.join()
  # p2.join()
  # p3.join()