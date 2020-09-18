# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import warnings
# import math
# from sklearn.linear_model import LinearRegression, Ridge, Lasso 
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
# from sklearn.feature_selection import SelectFromModel
# from sklearn.cluster import KMeans
# global alphas 

# alphas= np.linspace(0.1,1,num=50)

# def normalize(X):

# 	mean = [0]*X.shape[1]
# 	minimum = [10000000]*X.shape[1]
# 	Xnorm = X.copy()
# 	maximum = [-10000000]*X.shape[1]

# 	for i in range(X.shape[0]):
		
# 		for j in range(1, X.shape[1]):
			
# 			mean[j] += X[i][j]/X.shape[0]
# 			check=X[i][j]
# 			if (check > maximum[j]):
# 				maximum[j] =check
# 			if (check < minimum[j]):
# 				minimum[j] = check

# 	for i in range(X.shape[0]):
# 		for j in range(1, X.shape[1]):
# 			temp1=maximum[j] - minimum[j]
# 			check1=Xnorm[i][j]
# 			Xnorm[i][j] = check1- mean[j]/(temp1)

# 	return (mean, minimum, maximum, Xnorm)

# def gradientDescent(X, Y, start, end, learning_rate = 0.46, num_iterations=1000):
	
	
# 	RMS_training = []
# 	RMS_validation = []

# 	RMS1 = 0 
# 	RMS2 = 0 

# 	thetas = np.zeros([1, X.shape[1]])
# 	thetas_temp = np.zeros([1, X.shape[1]])


# 	iterations=0
# 	while iterations<(num_iterations):
# 		for j in range(X.shape[1]):
# 			thetas_temp[0][j] = thetas[0][j]
	
		
# 		for i in range(X.shape[0]):
# 			if (i >= start and i < end):
# 				continue
# 			else:
# 				for j in range(X.shape[1]):
# 					thetas_temp[0][j] -= ((learning_rate*((np.dot(thetas[0], X[i])) - Y[i]))/(X.shape[0] - (end - start)))*X[i][j]
# 		for j in range(X.shape[1]):
# 			thetas[0][j] = thetas_temp[0][j]
# 		for i in range(X.shape[0]):
# 			if (i < start or i >= end):
# 				temp1=np.dot(thetas[0], X[i])
# 				temp2=Y[i]
# 				RMS2 += (temp1-temp2)**2 
# 			else:
# 				temp1=np.dot(thetas[0], X[i])
# 				temp2=Y[i]
# 				RMS1 += (temp1-temp2)**2
# 		if end==start:
# 			pass
# 		else:
# 			RMS1 /= (2*(end - start))
# 			RMS2 /= (2*(X.shape[0] - (end - start)))
# 			RMS_training.append(math.sqrt(RMS2))
# 			RMS_validation.append(math.sqrt(RMS1))
# 			iterations+=1
# 	return	(RMS_training, RMS_validation)
				
# def gradientDescentRidge(X, Y, start, end, ridge, learning_rate = 0.46, num_iterations=1000):

	
# 	RMS_validation = []
# 	RMS_training = []
# 	RMS1 = 0
# 	RMS2 = 0
# 	thetas = np.zeros([1, X.shape[1]])
# 	thetas_temp = np.zeros([1, X.shape[1]])

# 	iterations=0

# 	while iterations<(num_iterations):
# 		for j in range(X.shape[1]):
# 			if (j == 0):
# 				pass
# 			else:
# 				temp1=ridge*thetas[0][j]
# 				temp2=X.shape[0] - end - start
# 				thetas_temp[0][j] = thetas[0][j] - temp1/temp2
# 		for i in range(X.shape[0]):
# 			if (i >= start and i < end):
# 				continue
# 			else:
# 				for j in range(X.shape[1]):
# 					thetas_temp[0][j] -= (learning_rate*((np.dot(thetas[0], X[i])) - Y[i]))/(X.shape[0] - (end - start))*X[i][j]

# 		for j in range(X.shape[1]):
# 			thetas[0][j] = thetas_temp[0][j]


		
# 		for i in range(X.shape[0]):
# 			if (i < start or i >= end):
# 				temp1=np.dot(thetas[0], X[i])
# 				temp2=Y[i]
# 				RMS2 += (temp1-temp2)**2
# 			else:
# 				temp1=np.dot(thetas[0], X[i])
# 				temp2=Y[i]
# 				RMS1 +=  (temp1-temp2)**2

# 		for j in range (1, X.shape[1]):
# 			temp1=math.pow(thetas[0][j],2)
# 			RMS1 += ridge*temp1
# 			temp2=math.pow(thetas[0][j],2)
# 			RMS2 += ridge*temp2
# 		if end==start:
# 			pass
# 		else:
# 			RMS1 /= (2*(end - start))
# 			RMS2 /= (2*(X.shape[0] - (end - start)))
# 			RMS_training.append(RMS2)
# 			RMS_validation.append(RMS1)
# 			iterations+=1
# 	return	(RMS_training, RMS_validation)

# def gradientDescentLasso(X, Y, start, end, lasso, learning_rate = 0.46, num_iterations=1000):

# 	thetas = np.zeros([1, X.shape[1]])
# 	thetas_temp = np.zeros([1, X.shape[1]])

# 	RMS_training = []
# 	RMS_validation = []
# 	RMS1 = 0
# 	RMS2 = 0

# 	iterations=0
# 	while iterations<(num_iterations):
# 		# Storing the values of Current theta into temporary variable
# 		# Also subtracting the Penalty for L2 Regularization
# 		for j in range(X.shape[1]):
# 			if (j == 0):
# 				pass			
# 			else:
# 				if (thetas[0][j] < 0) :
# 					temp1=2*(X.shape[0] - (end - start))
# 					thetas_temp[0][j] = thetas[0][j] + lasso/temp1
# 				else:
# 					temp1=2*(X.shape[0] - (end - start))
# 					thetas_temp[0][j] = thetas[0][j] - lasso/temp1
# 		for i in range(X.shape[0]):
# 			if (i >= start and i < end):
# 				continue
# 			else:	
# 				for j in range(X.shape[1]):
# 					thetas_temp[0][j] -= ((learning_rate*((np.dot(thetas[0], X[i])) - Y[i]))/(X.shape[0] - (end - start)))*X[i][j]
		
# 		for j in range(X.shape[1]):
# 			thetas[0][j] = thetas_temp[0][j]


# 		for i in range(X.shape[0]):
# 			if (i < start or i >= end):
# 				temp1=np.dot(thetas[0], X[i])
# 				temp2=Y[i][0]
# 				RMS2 += math.pow(temp1-temp2,2)
# 			else:
# 				temp1=np.dot(thetas[0], X[i])
# 				temp2=Y[i][0]
# 				RMS1 += (temp1-temp2)**2
		
# 		for j in range (1, X.shape[1]):
# 			temp1=abs(thetas[0][j])
# 			RMS1 += lasso*(temp1)
# 			RMS2 += lasso*(temp1)
# 		if end==start:
# 			pass
# 		else:
# 			RMS1 /= (2*(end - start))
# 			RMS2 /= (2*(X.shape[0] - (end - start)))
# 			RMS_training.append(RMS2)
# 			RMS_validation.append(RMS1)

# 			iterations+=1
		
# 	return	(RMS_training, RMS_validation)

# def gradientDescentkFold(X, Y):
	
# 	testing_range = X.shape[0]//5
# 	RMS_training = []
# 	mean_RMS_training = []
# 	mean_RMS_testing = []
# 	iterations = []
# 	RMS_validation = []
# 	rms = 0
# 	i=0
# 	temp1=X.shape[0]
# 	while i<5:
# 		if (i != 4):
# 			temp1=(i + 1)*testing_range
# 			temp2=i*testing_range
# 			rms = gradientDescent(X, Y,temp2,temp1 )
# 			RMS_training.append(rms[0])
# 			RMS_validation.append(rms[1])
# 		else:

# 			temp2=i*testing_range
# 			rms = gradientDescent(X,Y, temp2, temp1)
# 			RMS_training.append(rms[0])
# 			RMS_validation.append(rms[1])
# 		i+=1


# 	for i in range(len(RMS_training[0])):
# 		train = 0
# 		test = 0
# 		for j in range (5):
# 			temp1= RMS_training[j][i]
# 			train +=temp1/5
# 			temp2=RMS_validation[j][i]
# 			test += temp2/5
# 		mean_RMS_testing.append(test)
# 		mean_RMS_training.append(train)
# 		iterations.append(i + 1) 
	
# 	df=pd.DataFrame({'x': iterations, 'training_cost': mean_RMS_training, 'testing_cost': mean_RMS_testing})
# 	plt.xlabel('Iterations')
# 	plt.ylabel('Cost')
# 	plt.plot('x', 'training_cost', data=df)
# 	plt.plot('x', 'testing_cost', data=df)
# 	plt.legend()
# 	plt.title(title)
# 	plt.show()
# def choosemodel(X,Y,model):

# 	start  = (X.shape[0]//5)*1
# 	end  = (X.shape[0]//5)*2
	
# 	trainingX = np.vstack([X[0:start], X[end:]])
# 	testingX = np.vstack([X[start:end]])
	
# 	trainingY = np.vstack([Y[0:start], Y[end:]])
# 	testingY = np.vstack([Y[start:end]])

# 	model = Ridge()
# 	if model==1:
# 		grid = GridSearchCV(estimator=Ridge, param_grid=dict(alpha=alphas), cv=5)
# 		grid.fit(trainingX, trainingY)
# 		best=grid.best_estimator_.alpha
# 		temp = gradientDescentRidge(X, Y, start, end,best )
# 		iterations = []
# 		for i in range(len(temp[0])):
# 			iterations.append(i + 1)
		
# 	else:
# 		grid = GridSearchCV(estimator=Lasso, param_grid=dict(alpha=alphas), cv=5)
# 		grid.fit(trainingX, trainingY)
# 		best=grid.best_estimator_.alpha
# 		temp = gradientDescentLasso(X, datasetY, start, end,best)
# 		iterations = []
# 		for i in range(len(temp[0])):
# 			iterations.append(i + 1)
		



# X = np.load('X.npy')
# Y=np.load('Y.npy')
# normalize_mean, normalize_min, normalize_max, normalized_X = normalize(X)
# gradientDescentkFold(normalized_X, Y)
# choosemodel(normalized_datasetX, Y,1)
# choosemodel(normalized_datasetX, Y,2)