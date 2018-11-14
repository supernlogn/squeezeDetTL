import sys
import json
import os

# The currect directory, this file is in
BASE_PATH = os.path.dirname(__file__)

# Extensions of files found throughout the code
EXTENSIONS = [".py", ".pyx", ".json", ".c", ".cpp", ".h", ".sh"]

# Directories that contain source code files
DIRECTORIES = [".", "src", "scripts"]

# Map to replace all strings inside the directory on the left with those on the right.
REPLACEMENT_MAP = {
  # full path to your trainings folder, where you put directories for each training
  "/media/terabyte/projects/Thesis/trainings":  "/mnt/terabyte/sniper_trainings",
  # specify if you want GPU which GPU or if you want CPU and which CPU
  "tf.device('/job:localhost/replica:0/task:0/device:CPU:0')": "tf.device('/GPU:{}'.format(gpu_id))",
  # full path to your datasets folder, where you keep all your datasets organised
  "/media/terabyte/projects/datasets/" : "/mnt/terabyte/datasets/",
  # full path to the SqueezeNet weights trained in ImageNet.
  "/media/terabyte/projects/Thesis/SqueezeNet_imageNet_trained/" : "/mnt/terabyte/sniper_data/SqueezeNet_imageNet_trained/",
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