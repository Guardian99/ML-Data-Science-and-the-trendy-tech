import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import math
import copy
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectFromModel
from sklearn.cluster import KMeans
# ======================================================================
def gradientDescent(X, Y, learning_rate = 0.5, max_iter=1000,normal=0, lasso = 0, ridge = 0):
	RMS = 0
	thetas = np.zeros([1, X.shape[1]])
	iterations=0
	thetas_temp = np.zeros([1, X.shape[1]])
	while iterations<(max_iter):
		for j in range(X.shape[1]):
			thetas_temp[0][j] = thetas[0][j]
			if (j == 0):
				pass
			else:
				if (thetas_temp[0][j] > 0):
					check1=(1*lasso + thetas[0][j]*2*ridge)
					thetas_temp[0][j] -= check1/(2*X.shape[0])
				else:
					check2=((-1)*lasso + 2*ridge*thetas[0][j])
					thetas_temp[0][j] -= check2/(2*X.shape[0])
		for i in range(X.shape[0]):
			for j in range(X.shape[1]):
				thetas_temp[0][j] -= ((learning_rate*((np.dot(thetas[0], X[i])) - Y[i]))/(X.shape[0]))*X[i][j]
		for j in range(X.shape[1]):
			thetas[0][j] = thetas_temp[0][j]
		i=0
		while (i<X.shape[0]):
			temp1=np.dot(thetas[0], X[i])
			temp2=temp1- Y[i]
			RMS += math.pow(temp2,2)
			i+=1
		for j in range(1, X.shape[1]):
			temp=(thetas[0][j])**2
			check=(ridge*(temp) + lasso*(abs(thetas[0][j])))
			RMS += check/2*X.shape[0]
		RMS /= (2*X.shape[0])
		iterations+=1
	if normal==1:
		return RMS, thetas
	if lasso!=0:
		return RMS,thetas
	if ridge!=0:
		return RMS,thetas

df=pd.read_csv('data.csv')
X = df['Brain_Weight'].values
Y = df['Body_Weight'].values
x1=np.c_[np.ones(X.shape[0]),X]
# print(x1.shape)
mean = [0]*x1.shape[1]
maximum = [-10000000]*x1.shape[1]
minimum = [10000000]*x1.shape[1]
x1norm = x1.copy()
i=0
while i <(x1.shape[0]):
	# only the column with actual values
	j=1
	mean[j] += x1norm[i][j]/x1.shape[0]
	check=x1norm[i][j]
	if (x1norm[i][j] > maximum[j]):
		maximum[j] = check
	if (x1norm[i][j] < minimum[j]):
		minimum[j] =check
	i+=1

k=0
while k<x1.shape[0]:
	j=1
	x1norm[k][j] = x1norm[k][j] - mean[j]
	check1=maximum[j] - minimum[j]
	x1norm[k][j] = x1norm[k][j]/(check1)
	k+=1
# print(x1norm)
def graph(choice):
	if choice==1:
		rms, theta= gradientDescent(x1norm, Y,normal=1)
		check5=(maximum[1] - minimum[1])
		theta2 = theta[0][1]/check5
		theta1 = theta[0][0] - theta2*mean[1]
		plt.scatter(X, Y)
		axes = plt.gca()
		x_vals = np.array(axes.get_xlim())
		y_vals = theta1 + theta2 * x_vals
		plt.plot(x_vals, y_vals)
		plt.title('Scatter plot for Normal Gradient Descent')
		plt.xlabel('Brain Weight')
		plt.ylabel('Body Weight')
		plt.show()
	elif choice==2:
		rms, theta= gradientDescent(x1norm, Y,ridge=0.1)
		check5=(maximum[1] - minimum[1])
		theta2 = theta[0][1]/check5
		theta1 = theta[0][0] - theta2*mean[1]
		plt.scatter(X, Y)
		axes = plt.gca()
		x_vals = np.array(axes.get_xlim())
		y_vals = theta1 + theta2 * x_vals
		plt.plot(x_vals, y_vals)
		plt.title('Scatter plot for Ridge Gradient Descent')
		plt.xlabel('Brain Weight')
		plt.ylabel('Body Weight')
		plt.show()
	else:
		rms, theta= gradientDescent(x1norm, Y,lasso=0.7)
		check5=(maximum[1] - minimum[1])
		theta2 = theta[0][1]/check5
		theta1 = theta[0][0] - theta2*mean[1]
		plt.scatter(X, Y)
		axes = plt.gca()
		x_vals = np.array(axes.get_xlim())
		y_vals = theta1 + theta2 * x_vals
		plt.plot(x_vals, y_vals)
		plt.title('Scatter plot for Lasso Gradient Descent')
		plt.xlabel('Brain Weight')
		plt.ylabel('Body Weight')
		plt.show()
print('Enter choice for Gradient Descent')
print('1.Normal 2.RidgeRegularized 3.LassoRegularized' )
input1=int(input())
if input1==1:
	graph(1)
elif input1==2:
	graph(2)
else:
	graph(3)

# print(x1)
# ======================================================================

# ==========================================================