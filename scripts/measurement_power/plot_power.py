import numpy as np
import json
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path

if __name__ == "__main__":
  # kitti new
  fig = plt.figure()
  with open("kitti_new_power_log.json") as f:
    measurements = json.load(f)["gpu_1"]
  filtered_measurements = [m for m in measurements if m !=-1]
  n, bins = np.histogram(filtered_measurements)
  # get the corners of the rectangles for the histogram
  left = np.array(bins[:-1])
  right = np.array(bins[1:])
  bottom = np.zeros(len(left))
  top = bottom + n
  XY = np.array([[left, left, right, right], [bottom, top, top, bottom]]).T

  ax = plt.subplot(211)
  # get the Path object
  barpath = path.Path.make_compound_path_from_polys(XY)
  # make a patch out of it
  patch = patches.PathPatch(barpath)
  ax.add_patch(patch)
  # update the view limits
  ax.set_xlim(left[0], right[-1])
  ax.set_ylim(bottom.min(), top.max())  
  ax.grid(which='both')
  ax.legend(["new power consumption"], loc='upper center', shadow=True, fontsize='x-large')
  plt.xlabel('Watts')
  # kitti original
  with open("kitti_old_power_log.json") as f:
    measurements = json.load(f)["gpu_1"]
  filtered_measurements = [m for m in measurements if m !=-1]

  n, bins = np.histogram(filtered_measurements)
  # get the corners of the rectangles for the histogram
  left = np.array(bins[:-1])
  right = np.array(bins[1:])
  bottom = np.zeros(len(left))
  top = bottom + n
  XY = np.array([[left, left, right, right], [bottom, top, top, bottom]]).T

  ax = plt.subplot(212)
  # get the Path object
  barpath = path.Path.make_compound_path_from_polys(XY)
  # make a patch out of it
  patch = patches.PathPatch(barpath)
  ax.add_patch(patch)
  # update the view limits
  ax.set_xlim(left[0], right[-1])
  ax.set_ylim(bottom.min(), top.max())  
  ax.grid(which='both')
  ax.legend(["old power consumption"], loc='upper center', shadow=True, fontsize='x-large')
  plt.xlabel('Watts')
  plt.tight_layout()

  plt.savefig("power_compare.pdf")

  # total loss and mAP
  total = np.array(json.load(open("kitti_new_total_loss_1.json")))
  steps = total[:,1]
  mAps_raw = np.array(json.load(open("prediction_boxes_raw.txt")))
  mAPs, mAP_steps = [], []
  for r in mAps_raw:
      mAP_steps.append(r["step"])
      mAPs.append(r["mAP"])
  mAP_steps = np.array(mAP_steps)
  mAPs = np.array(mAPs)

  fig = plt.figure()
  plt.title('Train loss and mAP')
  ax = plt.subplot(121)
  plt.xlabel('steps')
  plt.ylabel('Total loss', fontsize=18)
  major_ticks = np.arange(0, 30, 3.0)
  ax.set_yticks(major_ticks)
  ax.grid(which='both')
  ax.plot(steps, total[:,-1], linewidth=2, color='tab:orange')
  ax = plt.subplot(122)
  # ax.set_aspect('equal')
  plt.xlabel('steps')
  plt.ylabel('mAP', fontsize=18)
  ax.grid(which='both')
  ax.plot(mAP_steps, mAPs, linewidth=2)
  plt.tight_layout()
  plt.savefig("Atrain.pdf", orientation="landscape")

  # step per time
  total_new = np.array(json.load(open("kitti_bench_total_loss.json")))
  new_steps = total_new[:,1]
  total_old = np.array(json.load(open("kitti_old_total_loss_1.json")))
  # print(L)  
  L = 250
  old_steps = total_old[:L,1]
  fig = plt.figure()
  ax = plt.subplot(111)
  plt.title("time/step comparison")
  # ax = plt.subplot(121)
  plt.xlabel('steps')
  plt.ylabel('total_time')
  ax.plot(new_steps, total_new[:,0] - total_new[0,0], linewidth=2, color='tab:green')
  ax.plot(old_steps, total_old[:L,0] - total_old[0,0], linewidth=2, color='xkcd:magenta')
  plt.xticks(np.arange(0, new_steps[-1] + 10000, step=5000))
  plt.grid(True)
  plt.legend("new", "old")
  pend = (new_steps[-1], np.max(total_new[:,0] - total_new[0,0]))
  pstart = (pend[0]-100, pend[1]-1000) 
  ax.annotate('(%d,%.2f)'%pend , xycoords='data', textcoords='data', xy=pend, xytext=pstart,
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"),
            )
  pend = (new_steps[-1], total_old[L,0] - total_old[0,0])
  pstart = (pend[0]-100, pend[1]-1000)
  ax.annotate('(%d,%.2f)'%pend , xycoords='data', textcoords='data', xy=pend, xytext=pstart,
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"))
  plt.savefig("kitti_compare.pdf")
  plt.show()
  
