import os
import sys
import numpy as np
import json
from multiprocessing import Process
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "src"))

from matplotlib import pyplot as plt
from matplotlib.patches import Polygon

# present two images as those of Bengio --> OK
# present the times of each run in one image
# pick resulting image detections --> possible

def read_results(BASE_DIR):
  # read B 
  B_ext_config = json.load(open("B_pascal_voc_config.json"))
  B_TRAIN_DIR = B_ext_config["TRAIN_DIR"]
  file_to_read = os.path.join(B_TRAIN_DIR, "evals", "mAP_history.json")
  with open(file_to_read) as f:
    # lines = f.readlines()
    # maps = [float(l.split("\"mAP\": ")[-1]) for l in lines if "mAP" in l]
    maps = [h["mAP"] for h in json.load(f)["history"]]
  B_mAP = max(maps)
  print(B_mAP)

  AnB, AnB_plus, BnB, BnB_plus = [B_mAP], [B_mAP], [B_mAP], [B_mAP]
  for i in range(11):
    # AnB
    file_to_read = os.path.join(BASE_DIR, "AnB_layer_%d"%i, "evals", "mAP_history.json")
    with open(file_to_read) as f:
      # lines = f.readlines()
      # maps = [float(l.split("\"mAP\": ")[-1]) for l in lines if "mAP" in l]
      maps = [h["mAP"] for h in json.load(f)["history"]]
    AnB.append(max(maps))
    # AnB+
    file_to_read = os.path.join(BASE_DIR, "AnB_plus_layer_%d"%i, "evals", "mAP_history.json")
    with open(file_to_read) as f:
      # lines = f.readlines()
      # maps = [float(l.split("\"mAP\": ")[-1]) for l in lines if "mAP" in l]
      maps = [h["mAP"] for h in json.load(f)["history"]]
    AnB_plus.append(max(maps))
    # BnB
    file_to_read = os.path.join(BASE_DIR, "BnB_layer_%d"%i, "evals", "mAP_history.json")
    with open(file_to_read) as f:
      # lines = f.readlines()
      # maps = [float(l.split("\"mAP\": ")[-1]) for l in lines if "mAP" in l]
      maps = [h["mAP"] for h in json.load(f)["history"]]
    BnB.append(max(maps))
    # BnB+
    file_to_read = os.path.join(BASE_DIR, "BnB_plus_layer_%d"%i, "evals", "mAP_history.json")
    with open(file_to_read) as f:
      # lines = f.readlines()
      # maps = [float(l.split("\"mAP\": ")[-1]) for l in lines if "mAP" in l]
      maps = [h["mAP"] for h in json.load(f)["history"]]
    BnB_plus.append(max(maps))

  print(AnB, AnB_plus, BnB, BnB_plus)
  with open("TLresults.txt", "w") as f:
    json.dump({"AnB" : AnB,
             "AnB+": AnB_plus,
             "BnB" : BnB,
             "BnB+": BnB_plus}, f)


def plot_results():
  # plot the lines
  # in our case we did that on a separate computer
  with open("TLresults.txt") as f:
    TLresults = json.load(f)
    AnB = TLresults["AnB"]
    AnB_plus = TLresults["AnB+"]
    BnB = TLresults["BnB"]
    BnB_plus = TLresults["BnB+"]
  X = np.arange(0,12,1)
  Ymax = AnB[0]
  fig = plt.figure()
  ax = plt.subplot(111)
  ax.plot(AnB, linewidth=2, color="r", label=r'$AnB$')
  verts = [(0, Ymax)] + list(zip(X, AnB)) + [(11, Ymax)]
  ax.add_patch(Polygon(verts, facecolor='xkcd:coral', edgecolor='1.0', alpha=0.3))
  ax.plot(AnB_plus, linewidth=2, color="xkcd:orange red", label=r'$AnB^+$')
  verts = [(0, Ymax)] + list(zip(X, AnB_plus)) + [(11, Ymax)]
  ax.add_patch(Polygon(verts, facecolor='xkcd:salmon', edgecolor='1.0', alpha=0.3))
  ax.plot(BnB, linewidth=2, color="b", label=r'$BnB$')
  verts = [(0, Ymax)] + list(zip(X, BnB)) + [(11, Ymax)]
  ax.add_patch(Polygon(verts, facecolor='xkcd:light blue', edgecolor='1.0', alpha=0.7))
  ax.plot(BnB_plus, linewidth=2, color="xkcd:sky blue", label=r'$BnB^+$')
  y = np.ones_like(AnB) * AnB[0]
  ax.plot(y, linestyle='--', linewidth=2, color="xkcd:black")
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  ax.legend()
  plt.xticks(np.arange(0,12,1))
  plt.xlabel(r"Layer $n$ at which network is chopped and retrained")
  plt.ylabel("mAP (higher is better)")
  plt.savefig("grantExp.pdf")

  # the same figure in logarithmic scale
  Ymax = np.log10(AnB[0])
  fig = plt.figure()
  ax = plt.subplot(111)
  ax.plot(np.log10(AnB), linewidth=2, color="r", label=r'$AnB$')
  verts = [(0, Ymax)] + list(zip(X, np.log10(AnB))) + [(11, Ymax)]
  ax.add_patch(Polygon(verts, facecolor='xkcd:coral', edgecolor='1.0', alpha=0.3))
  ax.plot(np.log10(AnB_plus), linewidth=2, color="xkcd:orange red", label=r'$AnB^+$')
  verts = [(0, Ymax)] + list(zip(X, np.log10(AnB_plus))) + [(11, Ymax)]
  ax.add_patch(Polygon(verts, facecolor='xkcd:salmon', edgecolor='1.0', alpha=0.3))
  ax.plot(np.log10(BnB), linewidth=2, color="b", label=r'$BnB$')
  verts = [(0, Ymax)] + list(zip(X, np.log10(BnB))) + [(11, Ymax)]
  ax.add_patch(Polygon(verts, facecolor='xkcd:light blue', edgecolor='1.0', alpha=0.3))
  ax.plot(np.log10(BnB_plus), linewidth=2, color="xkcd:sky blue", label=r'$BnB^+$')
  y = np.ones_like(AnB) * np.log10(AnB[0])
  ax.plot(y, linestyle='--', linewidth=2, color="xkcd:black")
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  ax.legend()
  plt.xticks(np.arange(0,12,1))
  plt.xlabel(r"Layer $n$ at which network is chopped and retrained")
  plt.ylabel(r"$\log_{10}(mAP)$ (higher is better)")
  plt.savefig("grantExpLog.pdf")
  plt.show()

if __name__ == "__main__":
  BASE_DIR = "/media/terabyte/projects/Thesis/trainings/TL_MINI_EXPERIMENT"
  # read_results(BASE_DIR)
  plot_results()
