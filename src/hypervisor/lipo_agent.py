import os
import dlib


def adalipo_search(mc, func, num_iterations, mins, maxs, is_integer, initial_values=[], history_file="hp_opt_history.txt"):
  """
  This function searches for the global maxima of func inside the
  parameters.
  Args:
    mc:
    func: Objective function to be optimized, defined as func(*args)
    num_iterations: Number of iterations till maxima is found
    mins[i]: minimum of hyperparameter #i to be optimized
    mins[i]: maximum of hyperparameter #i to be optimized
    is_integer[i]: if hyperparameter #i accepts only integer values
    initial_values: if this optimization has been done for some iterations
                    in the past then the results can bee added here as a list
                    with entries of type dlib.function_evaluation.
  Returns:
    The best values of the hyperparameters
    The best evaluation
  """
  specs = dlib.function_spec(mins, maxs, is_integer)

  if initial_values != []:
    opt = dlib.global_function_search([specs] * len(initial_values), initial_values, 0.001)
  else:
    opt = dlib.global_function_search(specs)
    
  for _ in xrange(num_iterations):
    function_evaluation_request_next = opt.get_next_x()
    x = function_evaluation_request_next.x
    f_val = func(*x)
    function_evaluation_request_next.set(f_val)
    with open(os.path.join(mc["BASE_DIR"], history_file), "a") as f:
      y = list(x)
      y.append(f_val)
      f.write(str(y))
      f.write("\n")

  # return best point so far
  r = opt.get_best_function_eval()
  opt_hp = list(r[0])
  opt_eval = r[1]
  return (opt_hp, opt_eval)
