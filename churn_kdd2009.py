# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 20:45:40 2018

@author: otavio
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

#leitura e dados
data = pd.read_csv("orange_small_train.data", sep='\t', header=None, dtype='unicode')
data_test = pd.read_csv("orange_small_test.data", sep='\t', header=None, dtype='unicode')
print(data.shape, data_test.shape)
num_features = list(data.columns[:190])
cat_features = list(data.columns[190:230])
data.head()
data_test.head()

def process_data(data, num_features, cat_features, data_test=None):
    delete_empty_features(data, data_test = data_test)
    process_num_features(data, num_features, data_test = data_test)
    process_cat_features(data, cat_features, data_test = data_test)
    #return data, data_test
    

def get_empty_features(data):
    empty_features = []
    c = 0.3
    for feat in data.columns:
        nulls = data[feat].isnull().value_counts()
        try:
            not_nulls = nulls[False]
            if not_nulls < c*40000:
                empty_features.append(feat)
        except:    
            empty_features.append(feat)
    print("numero de features vazias: ", len(empty_features))
    return empty_features

def delete_empty_features(data, data_test = None):
    #deletando features com mais de 70% dos valores vazios
    empty_features = get_empty_features(data)
    for feat in empty_features:
        if feat in num_features:
            num_features.remove(feat)
        else:
            cat_features.remove(feat)
        data.drop(feat, axis=1, inplace=True)
        if data_test is not None:
            data_test.drop(feat, axis=1, inplace=True)
    #return data, data_test


def process_cat_features(data, cat_features, data_test = None):
    #Processamento via hashing 
    hash_space = 50
    cat_x_hashed = pd.DataFrame()
    data_setA = [data]
    data_setB = [data_test]
    hash_set = [cat_x_hashed]

    for ftre in cat_features:
        for d, h in zip(data_setA, hash_set):
            ftre_hashed = [hash(x) % hash_space for x in d[ftre]]
            h[str(ftre)] = pd.Series(ftre_hashed)
    data[cat_features] = cat_x_hashed
    
    for ftre in cat_features:
        for d, h in zip(data_setB, hash_set):
            ftre_hashed = [hash(x) % hash_space for x in d[ftre]]
            h[str(ftre)] = pd.Series(ftre_hashed)
    data_test[cat_features] = cat_x_hashed
        
    #completando vazios com zeros
    data.fillna(0., inplace=True)
    if data_test is not None:
        data_test.fillna(0., inplace=True)
    #return data, data_test

def process_num_features(data, cat_features, data_test = None):
    #completando vazios com zeros
    for i,feat in enumerate(num_features):
        data_test = data_test.fillna(0)
    #return data, data_test

def scale(data, labels = False, data_test = None):
    scaler = StandardScaler()
    if labels:
        df = data.drop(labels = ['labels'], axis = 1)
        data_scaled = scaler.fit_transform(df)
        data_scaled = pd.DataFrame(data_scaled, index = df.index, columns = df.columns)
        data_scaled['labels'] = data['labels']     
    else: 
        data_scaled = scaler.fit_transform(data)
        data_scaled = pd.DataFrame(data_scaled, index = data.index, columns = data.columns)
        
    if data_test is not None:
        data_test_scaled = scaler.transform(data_test)
        data_test_scaled = pd.DataFrame(data_test_scaled, index = data_test.index, columns = data_test.columns)
    else:
        data_test_scaled = None
    return data_scaled, data_test_scaled

process_data(data, num_features, cat_features, data_test=data_test)
print(data.shape,data_test.shape)
data.head() 

data_scaled, data_test_scaled = scale(data, labels = True, data_test = data_test)
X_scaled = data_scaled.drop("labels", axis = 1)
y = data_scaled['labels']
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y, test_size=0.2, random_state=123)

classifiers = [
    DecisionTreeClassifier(),
    RandomForestClassifier(n_jobs = 2),
    GaussianNB()]

for classif in classifiers:
    roc_auc = cross_val_score(classif, X_train_scaled, y_train_scaled, cv=5, scoring='roc_auc')
    print("Métrica AUC para algoritmo %s é %f"%(classif.__class__.__name__, np.mean(roc_auc)))

clf = RandomForestClassifier(n_jobs = 2)
clf.fit(X_train_scaled, y_train_scaled)
y_pred = clf.predict_proba(X_test_scaled) 
y_prob = [x[1] for x in y_pred]
roc_auc_final = roc_auc_score(y_test_scaled, y_prob)
print('Métrica AUC no conjunto de dados é: ', roc_auc_final)

