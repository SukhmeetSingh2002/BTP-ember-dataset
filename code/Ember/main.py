import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

from data import data_loading
from data import data_visualization
from models import annModels
import tensorflow as tf


def handleANNModel():
  datasetDirectory = '../../filtered-data/standardizedv3'
  
  # X_train, X_test, Y_train, Y_test = data_loading.getTrainTestData(datasetDirectory,validation = False)
  X_train, X_val, X_test, Y_train, Y_val, Y_test = data_loading.getTrainTestData(datasetDirectory,validation = True)
  print()
  print(X_train.shape)
  print()
  print(tf.config.list_physical_devices('GPU'))
  model = annModels.getANN_1(X_train.shape[1])
  history = annModels.trainAnnModel(model, X_train, Y_train, X_val, Y_val, epochs = 50, batch_size = 2048)
  y_pred, y_pred_binary = annModels.predictAnnModel(model,X_test,Y_test,threshold = 0.5)
  data_visualization.plotAll(Y_test, y_pred, history, title = "Confusion Matrix", save = True)
  pass


def main():
  datasetDirectory = '../../filtered-data/standardizedv3'

  handleANNModel()
  
  
  pass
  

if __name__ == "__main__":
    main()