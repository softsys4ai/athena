"""
This is the script evaluating particular model on given data.
@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""

def eval_batch(model_name_list, data_file):
  """
  Evaluate a list of models using the same dataset.
  :param model_name_list: the list of models to evaluate.
  :param data_file: the data used to evaluate the models.
  :return: na
  """
  prefix, _, _, attack_info = data_file.split('-')

  scores = []
  for model_name in model_name_list:
    _, dataset, architect, trans_type = model_name.split('-')

    # load model
    pass

def eval_single(model):
  pass
