import os
import sys
import numpy as np
import json
from multiprocessing import Process
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "src"))

cwd = os.getcwd()

from auto_ITL import start


full_pascal_voc = {
  "CLASS_NAMES"           : ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
                              "car", "cat", "chair", "cow", "diningtable", "dog",
                              "horse", "motorbike", "person", "pottedplant", "sheep",
                              "sofa", "train", "tvmonitor"],
  "LABEL_INDICES"         : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
  "CLASSES"               : 20
}

freeze_layers = {
        "conv1": False,
        "fire2": False,
        "fire3": False,
        "fire4": False,
        "fire5": False,
        "fire6": False,
        "fire7": False,
        "fire8": False,
        "fire9": False,
        "fire10": False,
        "fire11": False,
        "conv12": False
}
hot_layers = {
        "conv1": False,
        "fire2": False,
        "fire3": False,
        "fire4": False,
        "fire5": False,
        "fire6": False,
        "fire7": False,
        "fire8": False,
        "fire9": False,
        "fire10": False,
        "fire11": False,
        "conv12": False
}


layers  = [
      "conv1",
      "fire2",
      "fire3",
      "fire4",
      "fire5",
      "fire6",
      "fire7",
      "fire8",
      "fire9",
      "fire10",
      "fire11",
      "conv12"
]

def main(TRAIN_DIR, num_classes=3):
  # create needed folders
  create_needed_dirs(TRAIN_DIR)

  # dataset A = half the pascal voc
  label_indices_A = np.array([1,2,3]) # for full pascal: np.random.choice(range(1,21), size=num_classes, replace=False)
  class_names_A = ["car", "pedestrian", "cyclist"] # for full pascal: full_pascal_voc["CLASS_NAMES"][label_indices_A - 1]
  config_A = json.load(open("/media/terabyte/projects/Thesis/transfer_learning/auto_ITL/scripts/kitti_tests/kitti_squeezeDet_config.json"))
  config_A["CLASS_NAMES"] = class_names_A
  config_A["CLASSES"] = num_classes
  config_A["LABEL_INDICES"] = list(label_indices_A)
  config_A["TRAIN_DIR"] = os.path.join(TRAIN_DIR, "A")
  A_ext_config = "/media/terabyte/projects/Thesis/transfer_learning/auto_ITL/scripts/tl_mini_experiment/A_config.json"
  json.dump(config_A, open(A_ext_config, "w"))
  # dataset B = the other half pascal voc
  label_indices_B = np.array([2,6,12])
  class_names_B = [full_pascal_voc["CLASS_NAMES"][i-1] for i in label_indices_B]
  config_B = json.load(open("/media/terabyte/projects/Thesis/transfer_learning/auto_ITL/scripts/tl_mini_experiment/full_pascal_voc_config.json"))
  config_B["CLASS_NAMES"] = class_names_B
  config_B["CLASSES"] = num_classes
  config_B["LABEL_INDICES"] = list(label_indices_B)
  config_B["TRAIN_DIR"] = os.path.join(TRAIN_DIR, "B")
  B_ext_config = "/media/terabyte/projects/Thesis/transfer_learning/auto_ITL/scripts/tl_mini_experiment/B_config.json"
  json.dump(config_B, open(B_ext_config, "w"))
  ### 1 st phase
  # model A
  # train model A in dataset A
  with open(os.path.join(config_A["TRAIN_DIR"], "init_info.txt"), "w") as f:
    f.write(str(label_indices_A) + "\n")
    f.write(str(class_names_A) + "\n")
  # start(A_ext_config)
  A_best_checkpoint = config_A["TRAIN_DIR"] # os.path.join(config_A["TRAIN_DIR"], max(os.listdir(config_A["TRAIN_DIR"]), key=os.path.getctime)) # the best is not always the latest, but...
  config_A["ckpt_path"] = A_best_checkpoint

  # model B
  # train model B in dataset B
  with open(os.path.join(config_B["TRAIN_DIR"], "init_info.txt"), "w") as f:
    f.write(str(label_indices_B) + "\n")
    f.write(str(class_names_B) + "\n")

  # start({"ext_config_file_path": B_ext_config})
  B_best_checkpoint = config_B["TRAIN_DIR"]# os.path.join(config_B["TRAIN_DIR"], max(os.listdir(config_B["TRAIN_DIR"]), key=os.path.getctime)) # the best is not always the latest, but...
  config_B["ckpt_path"] = B_best_checkpoint
  ### 2nd phase
  # models should load from checkpoints
  config_A["LOAD_PRETRAINED_MODEL"] = False
  config_A["DATASET_NAME"] = "PASCAL_VOC"
  config_A["DATA_PATH"] = config_B["DATA_PATH"]
  config_A["INITIAL_ANCHOR_SHAPES"] = config_B["INITIAL_ANCHOR_SHAPES"]
  config_A["CLASS_NAMES"] = class_names_B
  config_A["LABEL_INDICES"] = config_B["LABEL_INDICES"]
  config_A["REDUCE_DATASET"] = True
  config_A["PREPROCESSED_DATA_DIR"] = config_B["PREPROCESSED_DATA_DIR"]
  config_A["IMAGE_WIDTH"] = config_B["IMAGE_WIDTH"]
  config_A["IMAGE_HEIGHT"] = config_B["IMAGE_HEIGHT"]
  config_A["EVAL_ITERS"] = 200
  config_B["LOAD_PRETRAINED_MODEL"] = False
  LAYERS_TO_LOAD = []
  # Let a model C
  for i in range(11):
    freeze_layers[layers[i]] = True
    LAYERS_TO_LOAD.append(layers[i])
    # (AnB)
    # transfer from A to C all layers < i the rest are randomly initialized
    # freeze all layers < i of C
    # now train C in dataset B
    config_A["FREEZE_LAYERS"] = freeze_layers
    config_A["TRAIN_DIR"] = os.path.join(TRAIN_DIR, "AnB_layer_%d"%i)
    config_A["LAYERS_TO_LOAD"] = LAYERS_TO_LOAD

    A_ext_config = os.path.join(config_A["TRAIN_DIR"], "A_config.json")
    json.dump(config_A, open(A_ext_config, "w"))
    start({"ext_config_file_path": A_ext_config})
    # (AnB+)
    # transfer from A to C all layers < i the rest are randomly initialized
    # do not freeze any layer
    # now train C in dataset B
    config_A["FREEZE_LAYERS"] = hot_layers
    config_A["TRAIN_DIR"] = os.path.join(TRAIN_DIR, "AnB_plus_layer_%d"%i)
    config_A["LAYERS_TO_LOAD"] = LAYERS_TO_LOAD
    A_ext_config = os.path.join(config_A["TRAIN_DIR"], "A_config.json")
    json.dump(config_A, open(A_ext_config, "w"))
    start({"ext_config_file_path": A_ext_config})
    # (BnB)
    # transfer from B to C all layers < i the rest are randomly initialized
    # freeze all layers < i of C
    # now train C in dataset B
    config_B["FREEZE_LAYERS"] = freeze_layers
    config_B["TRAIN_DIR"] = os.path.join(TRAIN_DIR, "BnB_layer_%d"%i)
    config_B["LAYERS_TO_LOAD"] = LAYERS_TO_LOAD
    B_ext_config = os.path.join(config_B["TRAIN_DIR"], "B_config.json")
    json.dump(config_B, open(B_ext_config, "w"))
    start({"ext_config_file_path": B_ext_config})
    # (BnB+)
    # transfer from B to C all layers < i the rest are randomly initialized
    # do not freeze any layer
    # now train C in dataset B
    config_B["FREEZE_LAYERS"] = hot_layers
    config_B["TRAIN_DIR"] = os.path.join(TRAIN_DIR, "BnB_plus_layer_%d"%i)
    config_B["LAYERS_TO_LOAD"] = LAYERS_TO_LOAD
    B_ext_config = os.path.join(config_B["TRAIN_DIR"], "B_config.json")
    json.dump(config_B, open(B_ext_config, "w"))
    start({"ext_config_file_path": B_ext_config})
    # update that the i-th test has finished
    with open("report_status.txt", "a") as f:
      f.write(str(i) +"\n")

def create_needed_dirs(TRAIN_DIR):
  if(not os.path.exists(os.path.join(TRAIN_DIR, "A"))):
    os.mkdir(os.path.join(TRAIN_DIR, "A"))

  if(not os.path.exists(os.path.join(TRAIN_DIR, "B"))):
    os.mkdir(os.path.join(TRAIN_DIR, "B"))

  for i in range(12):
    if(not os.path.exists(os.path.join(TRAIN_DIR, "AnB_layer_%d"%i))):
      os.mkdir(os.path.join(TRAIN_DIR, "AnB_layer_%d"%i))
    if(not os.path.exists(os.path.join(TRAIN_DIR, "AnB_plus_layer_%d"%i))):
      os.mkdir(os.path.join(TRAIN_DIR, "AnB_plus_layer_%d"%i))
    if(not os.path.exists(os.path.join(TRAIN_DIR, "BnB_layer_%d"%i))):
      os.mkdir(os.path.join(TRAIN_DIR, "BnB_layer_%d"%i))
    if(not os.path.exists(os.path.join(TRAIN_DIR, "BnB_plus_layer_%d"%i))):
      os.mkdir(os.path.join(TRAIN_DIR, "BnB_plus_layer_%d"%i))

if __name__ == "__main__":
  TRAIN_DIR = "/media/terabyte/projects/Thesis/trainings/TL_MINI_EXPERIMENT"
  # for i in range(1, 5):
  main(os.path.join(TRAIN_DIR))




