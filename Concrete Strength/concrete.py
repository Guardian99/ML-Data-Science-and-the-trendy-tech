#general
import warnings
import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import eli5
import pickle
warnings.filterwarnings("ignore")
#===============================================================
#boost
import catboost as catb
import lightgbm as lgb
import xgboost
#===============================================================
#validation
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score,mean_squared_error
from sklearn.model_selection import KFold
#===============================================================
#model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,ExtraTreesRegressor,ExtraTreesClassifier,AdaBoostRegressor,AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
#===============================================================
#clustering
from sklearn.mixture import GaussianMixture
from sklearn.externals import joblib
from sklearn.cluster import KMeans
from nltk.cluster.kmeans import KMeansClusterer
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
#===============================================================
#dl
import tensorflow as tf
import keras
#===============================================================
#nlp
import nltk
import spacy
import gensim
#===============================================================
#scraping
import scrapy
import bs4
#===============================================================
#===============================================================
#===============================================================


df=pd.read_excel('Concrete_Data.xls')
df1=df.values
X=df1[:,0:8]
Y=df1[:,8]

Xtrain,XTest,Ytrain,YTest=train_test_split(X,Y,test_size=0.25,random_state=7)
rsme=[]


#===============================================================
# model1
model1=LinearRegression(fit_intercept=True, n_jobs=1, normalize=False)
model1.fit(Xtrain,Ytrain)
pred=model1.predict(XTest)
rsme.append(math.sqrt(mean_squared_error(pred,YTest)))

#===============================================================
# model2
model2=Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,normalize=False, random_state=None, solver='auto', tol=0.001)
model2.fit(Xtrain,Ytrain)
pred=model2.predict(XTest)
rsme.append(math.sqrt(mean_squared_error(pred,YTest)))
#===============================================================
# model3
model3=Lasso(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=1000,normalize=False, positive=False, precompute=False, random_state=None,selection='cyclic', tol=0.0001, warm_start=False)
model3.fit(Xtrain,Ytrain)
pred=model3.predict(XTest)
rsme.append(math.sqrt(mean_squared_error(pred,YTest)))
#===============================================================
# model4
model4=xgboost.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,max_depth=3, min_child_weight=1, missing=None, n_estimators=100,n_jobs=1, nthread=None, objective='reg:linear', random_state=0,reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,silent=True, subsample=1)
model4.fit(Xtrain,Ytrain)
pred=model4.predict(XTest)
rsme.append(math.sqrt(mean_squared_error(pred,YTest)))
#===============================================================
# model5
model5 = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,max_features='auto', max_leaf_nodes=None,min_impurity_decrease=0.0, min_impurity_split=None,min_samples_leaf=1, min_samples_split=2,min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,oob_score=False, random_state=None, verbose=0, warm_start=False)
model5.fit(Xtrain,Ytrain)
pred=model5.predict(XTest)
rsme.append(math.sqrt(mean_squared_error(pred,YTest)))
#===============================================================
# model6
model6=KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',metric_params=None, n_jobs=1, n_neighbors=5, p=2,weights='uniform')
model6.fit(Xtrain,Ytrain)
pred=model6.predict(XTest)
rsme.append(math.sqrt(mean_squared_error(pred,YTest)))
#===============================================================
# model7
model7=AdaBoostRegressor(base_estimator=None, learning_rate=1.0, loss='linear',n_estimators=50, random_state=None)
model7.fit(Xtrain,Ytrain)
pred=model7.predict(XTest)
rsme.append(math.sqrt(mean_squared_error(pred,YTest)))
#===============================================================
# model8
model8=ExtraTreesRegressor(bootstrap=False, criterion='mse', max_depth=None,max_features='auto', max_leaf_nodes=None,min_impurity_decrease=0.0, min_impurity_split=None,min_samples_leaf=1, min_samples_split=2,min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,oob_score=False, random_state=None, verbose=0, warm_start=False)
model8.fit(Xtrain,Ytrain)
pred=model8.predict(XTest)
rsme.append(math.sqrt(mean_squared_error(pred,YTest)))
#===============================================================
print(rsme)

filename = 'finalized_model.sav'
pickle.dump(model8, open(filename, 'wb'))