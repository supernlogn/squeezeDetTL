import sys
import json
import os

# full path to your trainings folder, where you put directories for each training
TRAININGS_FOLDER = "/mnt/terabyte/sniper_trainings"

# full path to your datasets folder, where you keep all your datasets organised
DATASETS_FOLDER = "/mnt/terabyte/datasets/"

# full path to the SqueezeNet weights pretrained in ImageNet classification.
PRETRAINED_WEIGHTS_FOLDER = "/mnt/terabyte/sniper_data/SqueezeNet_imageNet_trained/"

# The currect directory, this file is in
BASE_PATH = os.path.dirname(__file__)

# Extensions of files found throughout the code
EXTENSIONS = [".py", ".pyx", ".json", ".c", ".cpp", ".h", ".sh"]

# Directories that contain source code files
DIRECTORIES = [".", "src", "scripts"]


# Map to replace all strings inside the directory on the left with those on the right.
REPLACEMENT_MAP = {

  "/media/terabyte/projects/Thesis/trainings":  TRAININGS_FOLDER,
  # specify: whether you want GPU (then which GPU) or CPU (then which CPU)
  "tf.device('/job:localhost/replica:0/task:0/device:CPU:0')": "tf.device('/GPU:{}'.format(gpu_id))",
  
  "/media/terabyte/projects/datasets/" : DATASETS_FOLDER,
  
  "/media/terabyte/projects/Thesis/SqueezeNet_imageNet_trained/" : PRETRAINED_WEIGHTS_FOLDER,
  # full path to this folder
  "/media/terabyte/projects/Thesis/transfer_learning/auto_ITL/"  : BASE_PATH
}

def get_replaced_contents(file_path):
  """
    Args:
      Path to the file to have its contents replaced.
    Return:
      The new file's contents after replacement as a list of lines.
  """
  with open(file_path, "r") as f:
    new_lines = []
    for line in f:
      s = line
      for r_m in REPLACEMENT_MAP.keys():
        s = s.replace(r_m, REPLACEMENT_MAP[r_m])
      new_lines.append(s)
    return new_lines

def write_lines(file_path, r_c):
  with open(_path, "w") as f:
    f.writelines(r_c)

if __name__ == "__main__":
  for directory in DIRECTORIES:
    rootdir = os.path.join(BASE_PATH, directory) 
    for subdir, dirs, files in os.walk(rootdir):
      for filename in files:
        _path = os.path.join(BASE_PATH, subdir, filename)
        has_ext = any([filename.endswith(ext) for ext in EXTENSIONS])
        if (has_ext) and not _path == __file__:
          r_c = get_replaced_contents(_path)
          write_lines(_path, r_c)
