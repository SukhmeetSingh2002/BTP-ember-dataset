import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np


def extract_features(feature, feature_name, features):
    
    for key, value in feature.items(): 
        
        string_name = feature_name + '_' + key
        if string_name not in features:
            features[string_name]= []
        
        if isinstance(value, int) or isinstance(value, float):
            features[string_name].append(value)
        else:
            features[string_name].append(None)
    

def extract_features_from_dataset_to_dataframe(dataset):
    
    features= {}
    features['label']=[]

    section_names = ['.text', '.data', '.rsrc']
    feature_names = ['size', 'entropy', 'vsize']

    for section in section_names:
        for feature in feature_names:
            features[section +'_' + feature] = []

    features['machine']=[]
    features['subsystem']=[]


    for data in dataset:
        
        if data['label'] == -1:
            continue

        #all sections should be equal to 1    
        section_count_list = {'.text':0, '.data':0, '.rsrc':0}
        for item in data['section']['sections']:
            
            if item['name'] in section_names:
                section_count_list[item['name']]+=1
        
        check = False
        for key,value in section_count_list.items():
            if value!=1:
                check=True
                break
        
        if check:
            continue
            
        feature_names = ['size', 'entropy', 'vsize']
            
        for item in data['section']['sections']:

            if item['name'] in section_names:
                
                for feature in feature_names:
                    val = item[feature] if item[feature]!=None else None
                    features[item['name'] + '_' + feature].append(val)
        
        feature_names = ['general', 'strings']

        for feature in feature_names:
            extract_features(data[feature], feature,features)

        for item in data['header']:
            extract_features(data['header'][item], item,features)
        
        features['machine'].append(data['header']['coff']['machine'])
        features['subsystem'].append(data['header']['optional']['subsystem'])
        
        
        features['label'].append(data['label'])

    df = pd.DataFrame(features)
    print(df['label'].value_counts())
    return df


def extract_features_train_test(dataset_training,dataset_testing):
  training_data = extract_features_from_dataset_to_dataframe(dataset_training)
  testing_data = extract_features_from_dataset_to_dataframe(dataset_testing)
  return training_data,testing_data

def remove_null_columns(df_training,df_testing):
  df_training.dropna(how='all', axis=1, inplace=True)
  df_testing.dropna(how='all', axis=1, inplace=True)
  
  return df_training,df_testing


def label_encoding(df_training,df_testing,columns):
  column_data = {}
  for column in columns:
    column_data[column] = tuple(set(df_training[column].unique()).union(set(df_testing[column].unique())))
  
  le  = LabelEncoder()
  for column in columns:
    le.fit(column_data[column])
    df_training[column] = le.transform(df_training[column])
    df_testing[column] = le.transform(df_testing[column])
  
  return df_training,df_testing
    


def one_hot_encoding(df_training,df_testing,columns):
  df_training = pd.get_dummies(df_training, columns=columns)
  df_testing = pd.get_dummies(df_testing, columns=columns)
  
  for column in df_training.columns.difference(df_testing.columns):
    df_testing[column] = 0  
  for column in df_testing.columns.difference(df_training.columns):
    df_training[column]= 0
    
  df_testing = df_testing[df_training.columns] #ordering the columns
  
  return df_training,df_testing


def data_pruning(df_training,exclude_columns,threshold):
  for column in df_training.columns:

    if df_training[column].max()>10*df_training[column].median() and df_training[column].max()>10 and all([ not column.startswith(x) for x in exclude_columns]) :
        df_training[column] = np.where(df_training[column]<df_training[column].quantile(threshold), df_training[column], df_training[column].quantile(threshold))
  
  return df_training


def log_modification(df_training,df_testing):
  for column in df_training.columns:
    if df_training[column].nunique()>50 and df_training[column].max() > 10*df_training[column].median():
      print(column)
      df_training[column] = np.log(df_training[column]+1)
      df_testing[column] = np.log(df_testing[column]+1)  
  
  return df_training,df_testing
      

def standardize(df,df_train_fit):
    scaler = StandardScaler()
    exclude_columns= [
    "machine", "subsystem", "label"
    ]

    df_standardized = df.copy()

    for column in df.columns:
        if all([ not column.startswith(x) for x in exclude_columns]):
            # Standardize the column
            scaler = scaler.fit(df_train_fit[column].values.reshape(-1, 1))
            df_standardized[column] = scaler.transform(df[column].values.reshape(-1, 1))

    return df_standardized


def standardize_train_test(df_training,df_testing,exclude_coulumns):
  df_training_standardized = standardize(df_training,df_training)
  df_testing_standardized = standardize(df_testing,df_training)
  
  return df_training_standardized,df_testing_standardized
  
   
# traning_data = extract_features_from_dataset_to_dataframe(dataset_training)
# testing_data = extract_features_from_dataset_to_dataframe(dataset_testing)