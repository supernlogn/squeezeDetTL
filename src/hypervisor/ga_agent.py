import numpy as np
from six.moves import xrange

def gaussian(x, mu, sigma):
    return np.exp(-np.square(x - mu) / (2 * np.square(sigma)))


def translateDNA(pop, var_ranges):
  """
    pop: the population as 2D array of ints
    var_ranges: a dictionary from variables to lambdas with input int which return the value of the variable
  """
  _shape = np.shape(pop)
  population_vars = []
  for i in xrange(_shape[0]):
    var_row = {}
    for j, var_name in enumerate(var_ranges.keys()):
      var_row[var_name] = var_ranges[var_name](pop[i,j])
    population_vars.append(var_row)
  return population_vars

def get_population_values(func, translated_population, extra_configs):
  """
    Args:
      func: function to use for values
      translated: list of dictionaries with values
      extra_configs: extra list of non optimizable parameters to pass to func.
  """
  f_vals = []
  for i in xrange(len(translated_population)):
    f_input = extra_configs.copy()
    for key in translated_population[i]:
      f_input[key] = translated_population[i][key]
    f_vals.append(func(f_input)) # calculate the function
  return f_vals

def get_fittness(population_vals):
  return gaussian(np.array(population_vals), np.min(population_vals), 100.0)

def select(population, fittness, population_size):
  """
    The idea of selection phase is to select the fittest individuals
    and let them pass their genes to the next generation.
  """
  idx = np.unique(np.random.choice(a=np.arange(population_size), size = population_size, replace=True, p = fittness/fittness.sum()))
  return population[idx]

def crossover(parent, population, population_size, dna_size, cross_rate):
  if np.random.rand() < cross_rate:
    i_ = np.random.randint(0, population_size, size=1)                            # select another individual from pop
    cross_points = np.random.randint(0, 2, size=dna_size).astype(np.bool)         # choose crossover points
    parent[cross_points] = population[i_, cross_points]                           # mating and produce one child
  return parent

def mutate(child, dna_size, mutation_rate, var_ranges_dtype):
  for point in xrange(dna_size):
    if np.random.rand() < mutation_rate:
      child[point] = child[point] ^ np.random.randint(low=np.iinfo(var_ranges_dtype).min, high=np.iinfo(var_ranges_dtype).max, dtype=var_ranges_dtype)
  return child

# START
# Generate the initial population
# Compute fitness
# REPEAT
#     Selection
#     Crossover
#     Mutation
#     Compute fitness
# UNTIL population has converged
# STOP

def ga_search(func, var_ranges, population_size, cross_rate, mutation_rate, constant_configs={}, var_ranges_dtype=np.int16):
  dna_size = len(var_ranges.keys())
  population = np.random.randint(low=np.iinfo(var_ranges_dtype).min, high=np.iinfo(var_ranges_dtype).max, size=(population_size, dna_size), dtype=var_ranges_dtype)   # initialize the gene population
  all_time_min_vars = {var_name: var_ranges[var_name](0) for var_name in var_ranges.keys()}
  for s in constant_configs.keys():
    if s not in all_time_min_vars.keys():
      all_time_min_vars[s] = constant_configs[s]
  all_time_min_val = func(all_time_min_vars)
  while True:
    # generate population values
    translated_population = translateDNA(population, var_ranges)
    F_values = get_population_values(func, translated_population, constant_configs)
    if(np.min(F_values) < all_time_min_val):
      all_time_min_vars = translated_population[np.argmin(F_values)]
      all_time_min_val = np.min(F_values)
    yield (all_time_min_vars, all_time_min_val)
    # select
    population = select(population, get_fittness(F_values), population_size)
    population_size = np.shape(population)[0]
    pop_copy = population.copy()
    # crossover
    children = [crossover(parent, pop_copy, population_size, dna_size, cross_rate) for parent in population]
    # mutate
    mutants  = [mutate(child, dna_size, mutation_rate, var_ranges_dtype) for child in children]
    population = np.array(mutants)
  return