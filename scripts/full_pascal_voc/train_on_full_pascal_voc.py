# This is a script to train squeezeDet on full pascal voc
# and save the convergent trained checkpoint
# It is accompanied by a json configuration file
import os 
import sys
import json
import shutil
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "src"))

from auto_ITL import start

if __name__ == "__main__":
  args = {}
  dir_path = os.path.dirname(os.path.abspath(__file__))
  args["ext_config_file_path"] = os.path.join(dir_path, "train_on_full_pascal_voc2.json")
  with open(args["ext_config_file_path"], "r") as f:
    DIR_NAME = json.load(f)["BASE_DIR"]
  # recreate directory if it exists and create if it doesn't exist
  if(os.path.exists(DIR_NAME)):
    shutil.rmtree(DIR_NAME)
  os.makedirs(DIR_NAME)
  # wish good luck, cause it ain't gonna have one :P 
  start(args)
