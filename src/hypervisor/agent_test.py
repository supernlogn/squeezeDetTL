from lipo_agent import adalipo_search
from dlib import find_min_global as adalipo_search_v2
from random_agent import random_search
from ga_agent import ga_search
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# test 2D
def f(x):
  return (x[0] - 2)*(x[0] - 2)*(x[1] - 1)*(x[1] - 1)

def f_sep(x0, x1):
  return f([x0, x1])

def f_args(*x):
  return f([x[0],x[1]])

def f_labeled(x):
  return (x["0"] - 2)*(x["0"] - 2)*(x["1"] - 1)*(x["1"] - 1)

var_limits = {
  "0" : {"max":10.0, "min":-10.0},
  "1" : {"max":10.0, "min":-10.0}}

var_ranges = {
  "0" : lambda i: (var_limits["0"]["max"] - var_limits["0"]["min"]) * i / float(np.iinfo(np.int16).max),
  "1" : lambda i: (var_limits["1"]["max"] - var_limits["1"]["min"]) * i / float(np.iinfo(np.int16).max)}

X = np.reshape(np.stack(np.meshgrid(np.arange(-10,10,0.01), np.arange(-10,10,0.01)), axis=-1), [-1,2])
K = np.concatenate([np.arange(0.1,1.5, 0.01), np.power((1+0.5), np.arange(2.0, 49.0, 0.1))])

# v = adalipo_search(K, X, f, 0.5)
# for i in range(20):
#   print(v.next())
x, y = adalipo_search_v2(f_args, [-10.0, -10.0], [10.0, 10.0], 10)
print(x,y)

# v = random_search(f_labeled, var_ranges)
# for i in range(100):
#   print(v.next())

population_size = 100
cross_rate = 0.6
mutation_rate = 0.03


# v = ga_search(f_labeled, var_ranges, population_size, cross_rate, mutation_rate)
# fig = plt.figure()
# plt.ion()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(0.0, 0.0, 0.0, s=200, lw=0, c='blue', alpha=0.5); plt.pause(0.05)
# for i in range(200):
#   ri, rv = v.next()
#   print(rv, ri)
#   ax.scatter(ri["0"], ri["1"], rv, s=200, lw=0, c='red', alpha=0.5); plt.pause(0.05)
# plt.ioff(); plt.show()