import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



def load_dataset(fileName):
  dfTraining = pd.read_csv(fileName+ '/training.csv')
  dfTesting = pd.read_csv(fileName+ '/testing.csv')
  return dfTraining, dfTesting
  pass

def train_test_split_from_dataframe(df_training, df_testing):
    y_train = df_training['label']
    X_train = df_training.drop('label', axis=1)

    y_test = df_testing['label']
    X_test = df_testing.drop('label', axis=1)

    return X_train, X_test, y_train, y_test

def getTrainTestData(fileName,val_split = 0.2,validation = False):
  dfTraining, dfTesting = load_dataset(fileName) 
  X_train, X_test, Y_train, Y_test = train_test_split_from_dataframe(dfTraining, dfTesting)
  X_train = X_train.astype(np.float32)
  X_test = X_test.astype(np.float32)
  
  if validation:
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=val_split, random_state=42)
    X_val = X_val.astype(np.float32)
    return X_train, X_val, X_test, Y_train, Y_val, Y_test
  
  return X_train, X_test, Y_train, Y_test  