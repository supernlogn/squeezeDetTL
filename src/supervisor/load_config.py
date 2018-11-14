from json import load
from os.path import join, dirname

def super_config():
  super_config = load(open(join(dirname(__file__), 
                                "supervisor_config.json"), "r"))
  return super_config
  # for x in super_config:
  #   if(isinstance(super_config[x], basestring)):
  #     tf_flags.DEFINE_string(x, super_config[x], """ """)
  #   elif(isinstance(super_config[x], float)):
  #     tf_flags.DEFINE_float(x, super_config[x], """ """)
  #   elif(isinstance(super_config[x], int)):
  #     tf_flags.DEFINE_integer(x, int(super_config[x]), """ """)
  #   elif(isinstance(super_config[x], bool)):
  #     tf_flags.DEFINE_boolean(x, super_config[x], """ """)