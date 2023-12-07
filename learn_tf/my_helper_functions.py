#Function to evluate: aacuracy, precision, recall, f1-score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
def evaluate_prediction_results(y_true, y_pred):
  """
  Calculates model accuracy, precision, recall, and f1-score
  """
  model_accuracy = accuracy_score(y_true, y_pred)
  model_precision, model_recall, model_f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
  model_metrics = {'accuracy': model_accuracy*100,
                   'precision': model_precision*100,
                   'recall': model_recall*100,
                   'f1-score': model_f1_score*100
                   }
  return model_metrics

import datetime
import tensorflow as tf
def create_tensorboard_callback(dir_name, model_name):
  path_name = dir_name + "/" + model_name + "-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensor_board = tf.keras.callbacks.TensorBoard(log_dir=dir_name)
  print(f"Saving TensorBoard log files to {path_name}")
  return tensor_board