import numpy as np



def random_search(func, var_ranges, constant_configs={}, var_ranges_dtype=np.int16):
  """
  Args:
    func: function to find maxima
  """
  Xi = []
  fx = []
  while True:
    Xnew = {var_name: var_ranges[var_name](np.random.randint(low=np.iinfo(var_ranges_dtype).min, high=np.iinfo(var_ranges_dtype).max, dtype=var_ranges_dtype))
            for var_name in var_ranges.keys()}
    if not np.any([np.all(xi.values() == Xnew.values()) for xi in Xi]):
      Xi.append(Xnew)
      x = Xnew
      for s in constant_configs.keys():
        if s not in x.keys():
          x[s] = constant_configs[s]
      fx.append(func(x))
    else:
      continue
    yield (Xi[np.argmin(fx)], np.min(fx))
  return