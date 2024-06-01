import numpy as np
from sklearn.metrics import accuracy_score

def predict_model(model,X_test,y_test,encoder,print_metric = False):
  y_pred = model.predict(X_test)
  y_pred = np.argmax(y_pred, axis=1)
  y_pred_class = encoder.inverse_transform(y_pred)
  
  if print_metric:
    print("Accurary:",accuracy_score(y_test, y_pred))
  return y_pred, y_pred_class
