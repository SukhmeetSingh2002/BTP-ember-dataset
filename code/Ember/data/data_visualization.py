from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plotROC(y_test, y_pred,save = True):
  fpr, tpr, thresholds = roc_curve(y_test, y_pred)
  roc_auc = auc(fpr, tpr)
  # plt.figure()
  plt.figure(figsize=(8,5))
  plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.0])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver Operating Characteristic')
  plt.legend(loc="lower right")

  plt.show()
  # Calculate AUC score
  auc_score = roc_auc_score(y_test, y_pred)
  print("AUC Score:", auc_score)
  return

def plot_confusion_matrix(y_true, y_pred,title, labels=[0,1],save = True):
  cm = confusion_matrix(y_true, y_pred, labels=labels)
  df_cm = pd.DataFrame(cm, index=labels, columns=labels)
  plt.figure(figsize=(10, 7))
  sns.heatmap(df_cm, annot=True, fmt='g')
  # increase font size
  sns.set_theme(font_scale=2)
  plt.title(title)
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()
  return

def display_classification_report(y_test, y_pred):
  print(classification_report(y_test, y_pred))
  
def plotLossCurve(history,save = True):

  train_loss = history.history['loss']
  val_loss = history.history['val_loss']

  #plot graph
  # plt.figure().set_figwidth(7)
  plt.plot(train_loss, label='Training Loss')
  plt.plot(val_loss, label='Validation Loss')


  plt.legend()
  plt.show()
  
  
def plotAll(y_test, y_pred, y_pred_binary,history, title = "Confusion Matrix  ANN", labels=[0,1],save = True):
  plotROC(y_test, y_pred,save)
  plot_confusion_matrix(y_test, y_pred_binary,title,labels = labels, save=save)
  display_classification_report(y_test, y_pred_binary)
  plotLossCurve(history,save)
  return
 