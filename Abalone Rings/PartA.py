import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import math
from sklearn import preprocessing
# ===========================================================
# EDA-PART.....Ignore else run to see it in all its glory

df=pd.read_csv('abalone.csv')
# print(df.info())
# print(df['Sex'].value_counts())
# no missing values....3 sexes
# print(df.head(5))
sex = {'M': 1,'F': 2,'I':3} 
df['Sex'] = [sex[item] for item in df['Sex']]
# print(df.info())
# print(df['Sex'].value_counts())
# correlation=df.corr()
# correlation_values = correlation['Rings'].sort_values(ascending = False)
# print(correlation_values)

# scaling
sex=df['Sex']
diameter=preprocessing.scale(df['Diameter'])
height=preprocessing.scale(df['Height'])
length=preprocessing.scale(df['Length'])
shell_weight=preprocessing.scale(df['Shell weight'])
whole_weight=preprocessing.scale(df['Whole weight'])
viscera_weight=preprocessing.scale(df['Viscera weight'])
shucked_weight=preprocessing.scale(df['Shucked weight'])
# ===========================================================

X = np.c_[np.ones(diameter.shape[0]),sex,diameter,height,length,shell_weight,whole_weight,viscera_weight,shucked_weight]
np.save('X.npy',X)
# print(X)
Y=df['Rings'].values
np.save('Y.npy',Y)

# print(Y)
# print(len(Y))
# ===========================================================



def gradient_descent(x, y, m, theta, alpha):
    cost_list = [] 
    theta_list = [] 
    prediction_list = []
    flag = True
    cost_list.append(100000)    
    i=0
    while flag:
        prediction = np.dot(x, theta)
        prediction_list.append(prediction)
        error = prediction - y
        cost = 1/(2*m) * np.dot(error.T, error)
        cost_list.append(cost)
        theta = theta - (alpha * (1/m) * np.dot(x.T, error)) 
        theta_list.append(theta)
        
        # by how much should each cost differ so to proceed with gradient descent...if small then it would converge quickly and if very very very small then takes time 
        if cost_list[i]-cost_list[i+1] < 0.000001:
            flag = False

        i+=1
    cost_list.pop(0)
    return prediction_list, cost_list, theta_list

rsmetrainingaverage=[]
rsmevalidationaverage=[]

rsmetrainingaveragenormal=[]
rsmevalidationaveragenormal=[]
# ===========================================================

# cross-validation

# fold1
alpha = 0.001
sex1=sex[836:]
diameter1=diameter[836:]
height1=height[836:]
length1=length[836:]
shell_weight1=shell_weight[836:]
whole_weight1=whole_weight[836:]
viscera_weight1=viscera_weight[836:]
shucked_weight1=shucked_weight[836:]
y1=Y[836:]
x1 = np.c_[np.ones(diameter1.shape[0]),sex1,diameter1,height1,length1,shell_weight1,whole_weight1,viscera_weight1,shucked_weight1]
m = y1.size  
theta = np.random.rand(9)
prediction_list1, cost_list1, theta_list1 = gradient_descent(x1, y1, m, theta, alpha)
theta1 = theta_list1[-1]

predictions=x1.dot(theta1)
trainerrorfold1=y1-predictions
trainerrorgd1=np.square(trainerrorfold1)
sum1fromtrainerrorgd1=sum(trainerrorgd1)/len(trainerrorgd1)
rsmetrainerrorgd1=math.sqrt(sum1fromtrainerrorgd1)

rsmetrainingaverage.append(rsmetrainerrorgd1)
# print('trainerrorfold1=',rsmetrainerrorgd1)
# print(trainerrorfold1)
# print('theta_gdfold1=',theta1)

# normal equation
x_transpose = np.transpose(x1)   
transpose = x_transpose.dot(x1)  
temp_1 = np.linalg.inv(transpose) 
temp_2 = x_transpose.dot(y1) 
theta_normal = temp_1.dot(temp_2)

predictions=x1.dot(theta_normal)
trainerrorfold1=y1-predictions
trainerrornormal1=np.square(trainerrorfold1)
sum1fromtrainerrornormal1=sum(trainerrornormal1)/len(trainerrornormal1)
rsmetrainerrornormal1=math.sqrt(sum1fromtrainerrornormal1)
rsmetrainingaveragenormal.append(rsmetrainerrornormal1)

# print('theta_normalfold1=',theta_normal)

sexvalidate1=sex[0:836]
diametervalidate1=diameter[0:836]
heightvalidate1=height[0:836]
lengthvalidate1=length[0:836]
shell_weightvalidate1=shell_weight[0:836]
whole_weightvalidate1=whole_weight[0:836]
viscera_weightvalidate1=viscera_weight[0:836]
shucked_weightvalidate1=shucked_weight[0:836]
yvalidate1=Y[0:836]

xvalidate1= np.c_[np.ones(diametervalidate1.shape[0]),sexvalidate1,diametervalidate1,heightvalidate1,lengthvalidate1,shell_weightvalidate1,whole_weightvalidate1,viscera_weightvalidate1,shucked_weightvalidate1]
# print(y1)
# print(xvalidate1.dot(theta_normal))
predictionsfromgd=xvalidate1.dot(theta1)
differencegd=np.subtract(yvalidate1,predictionsfromgd)
errorgd1=np.square(differencegd)
sum1fromerrorgd1=sum(errorgd1)/len(errorgd1)
rsmeerrorgd1=math.sqrt(sum1fromerrorgd1)
rsmevalidationaverage.append(rsmeerrorgd1)
# print(rsmeerrorgd1)
predictionsfromnormal=xvalidate1.dot(theta_normal)
differencenormal=np.subtract(yvalidate1,predictionsfromnormal)
errornormal1=np.square(differencenormal)
sum1fromerrornormal1=sum(errornormal1)/len(errornormal1)
rsmeerrornormal1=math.sqrt(sum1fromerrornormal1)
rsmevalidationaveragenormal.append(rsmeerrornormal1)
# print(rsmeerrornormal1)

# ===========================================================

# fold2

alpha = 0.001
sex2=sex[0:len(Y)-836]
diameter2=diameter[0:len(Y)-836]
height2=height[0:len(Y)-836]
length2=length[0:len(Y)-836]
shell_weight2=shell_weight[0:len(Y)-836]
whole_weight2=whole_weight[0:len(Y)-836]
viscera_weight2=viscera_weight[0:len(Y)-836]
shucked_weight2=shucked_weight[0:len(Y)-836]
y2=Y[0:len(Y)-836]
x2 = np.c_[np.ones(diameter2.shape[0]),sex2,diameter2,height2,length2,shell_weight2,whole_weight2,viscera_weight2,shucked_weight2]
m = y2.size  
theta = np.random.rand(9)
prediction_list1, cost_list1, theta_list1 = gradient_descent(x2, y2, m, theta, alpha)
theta2 = theta_list1[-1]
# print('theta_gdfold1=',theta1)
predictions=x2.dot(theta2)
trainerrorfold2=y2-predictions
trainerrorgd2=np.square(trainerrorfold2)
sum1fromtrainerrorgd2=sum(trainerrorgd2)/len(trainerrorgd2)
rsmetrainerrorgd2=math.sqrt(sum1fromtrainerrorgd2)

rsmetrainingaverage.append(rsmetrainerrorgd2)
# normal equation
x_transpose = np.transpose(x2)   
transpose = x_transpose.dot(x2)  
temp_1 = np.linalg.inv(transpose) 
temp_2 = x_transpose.dot(y2) 
theta_normal2 = temp_1.dot(temp_2)
# print('theta_normalfold1=',theta_normal)

predictions=x2.dot(theta_normal2)
trainerrornormal2=y2-predictions
trainerrornormal2=np.square(trainerrornormal2)
sum1fromtrainerrornormal2=sum(trainerrornormal2)/len(trainerrornormal2)
rsmetrainerrornormal2=math.sqrt(sum1fromtrainerrornormal2)
rsmetrainingaveragenormal.append(rsmetrainerrornormal2)


sexvalidate2=sex[len(Y)-836:]
diametervalidate2=diameter[len(Y)-836:]
heightvalidate2=height[len(Y)-836:]
lengthvalidate2=length[len(Y)-836:]
shell_weightvalidate2=shell_weight[len(Y)-836:]
whole_weightvalidate2=whole_weight[len(Y)-836:]
viscera_weightvalidate2=viscera_weight[len(Y)-836:]
shucked_weightvalidate2=shucked_weight[len(Y)-836:]
yvalidate2=Y[len(Y)-836:]

xvalidate2= np.c_[np.ones(diametervalidate2.shape[0]),sexvalidate2,diametervalidate2,heightvalidate2,lengthvalidate2,shell_weightvalidate2,whole_weightvalidate2,viscera_weightvalidate2,shucked_weightvalidate2]
# print(y1)
# print(xvalidate1.dot(theta_normal))
predictionsfromgd=xvalidate2.dot(theta2)
differencegd=np.subtract(yvalidate2,predictionsfromgd)
errorgd2=np.square(differencegd)
sum1fromerrorgd2=sum(errorgd2)/len(errorgd2)
rsmeerrorgd2=math.sqrt(sum1fromerrorgd2)
rsmevalidationaverage.append(rsmeerrorgd2)

predictionsfromnormal=xvalidate2.dot(theta_normal2)
differencenormal=np.subtract(yvalidate2,predictionsfromnormal)
errornormal2=np.square(differencenormal)
sum1fromerrornormal2=sum(errornormal2)/len(errornormal2)
rsmeerrornormal2=math.sqrt(sum1fromerrornormal2)
rsmevalidationaveragenormal.append(rsmeerrornormal2)
# print(rsmeerrornormal2)


# =================================================================

# fold3

alpha = 0.001

templaal1=sex[0:836]
templaal2=sex[1672:]
sex3=np.append(templaal1,templaal2,axis=0)


templaal1=diameter[0:836]
templaal2=diameter[1672:]
diameter3=np.append(templaal1,templaal2,axis=0)

templaal1=height[0:836]
templaal2=height[1672:]
height3=np.append(templaal1,templaal2,axis=0)

templaal1=length[0:836]
templaal2=length[1672:]
length3=np.append(templaal1,templaal2,axis=0)

templaal1=shell_weight[0:836]
templaal2=shell_weight[1672:]
shell_weight3=np.append(templaal1,templaal2,axis=0)

templaal1=whole_weight[0:836]
templaal2=whole_weight[1672:]
whole_weight3=np.append(templaal1,templaal2,axis=0)

templaal1=viscera_weight[0:836]
templaal2=viscera_weight[1672:]
viscera_weight3=np.append(templaal1,templaal2,axis=0)

templaal1=shucked_weight[0:836]
templaal2=shucked_weight[1672:]
shucked_weight3=np.append(templaal1,templaal2,axis=0)

templaal1=Y[0:836]
templaal2=Y[1672:]
y3=np.append(templaal1,templaal2,axis=0)


x3 = np.c_[np.ones(diameter3.shape[0]),sex3,diameter3,height3,length3,shell_weight3,whole_weight3,viscera_weight3,shucked_weight3]
m = y3.size  
theta = np.random.rand(9)
prediction_list1, cost_list1, theta_list1 = gradient_descent(x2, y2, m, theta, alpha)
theta3 = theta_list1[-1]
# print('theta_gdfold1=',theta3)
predictions=x3.dot(theta3)
trainerrorfold3=y3-predictions
trainerrorgd3=np.square(trainerrorfold3)
sum1fromtrainerrorgd3=sum(trainerrorgd3)/len(trainerrorgd3)
rsmetrainerrorgd3=math.sqrt(sum1fromtrainerrorgd3)

rsmetrainingaverage.append(rsmetrainerrorgd3)
# normal equation
x_transpose = np.transpose(x3)   
transpose = x_transpose.dot(x3)  
temp_1 = np.linalg.inv(transpose) 
temp_2 = x_transpose.dot(y3) 
theta_normal3 = temp_1.dot(temp_2)
# print('theta_normalfold1=',theta_normal3)

predictions=x3.dot(theta_normal3)
trainerrornormal3=y3-predictions
trainerrornormal3=np.square(trainerrornormal3)
sum1fromtrainerrornormal3=sum(trainerrornormal3)/len(trainerrornormal3)
rsmetrainerrornormal3=math.sqrt(sum1fromtrainerrornormal3)
rsmetrainingaveragenormal.append(rsmetrainerrornormal3)



sexvalidate3=sex[836:1672]
diametervalidate3=diameter[836:1672]
heightvalidate3=height[836:1672]
lengthvalidate3=length[836:1672]
shell_weightvalidate3=shell_weight[836:1672]
whole_weightvalidate3=whole_weight[836:1672]
viscera_weightvalidate3=viscera_weight[836:1672]
shucked_weightvalidate3=shucked_weight[836:1672]
yvalidate3=Y[836:1672]
xvalidate3= np.c_[np.ones(diametervalidate3.shape[0]),sexvalidate3,diametervalidate3,heightvalidate3,lengthvalidate3,shell_weightvalidate3,whole_weightvalidate3,viscera_weightvalidate3,shucked_weightvalidate3]

# print(y1)
# print(xvalidate1.dot(theta_normal))

predictionsfromgd=xvalidate3.dot(theta3)
differencegd=np.subtract(yvalidate3,predictionsfromgd)
errorgd3=np.square(differencegd)
sum1fromerrorgd3=sum(errorgd3)/len(errorgd3)
rsmeerrorgd3=math.sqrt(sum1fromerrorgd3)
rsmevalidationaverage.append(rsmeerrorgd3)

predictionsfromnormal=xvalidate3.dot(theta_normal3)
differencenormal=np.subtract(yvalidate3,predictionsfromnormal)
errornormal3=np.square(differencenormal)
sum1fromerrornormal3=sum(errornormal3)/len(errornormal3)
rsmeerrornormal3=math.sqrt(sum1fromerrornormal3)
rsmevalidationaveragenormal.append(rsmeerrornormal3)
# print(rsmeerrornormal3)


# =================================================================

# fold4

alpha = 0.001

templaal1=sex[0:1673]
templaal2=sex[2506:]
sex4=np.append(templaal1,templaal2,axis=0)


templaal1=diameter[0:1673]
templaal2=diameter[2506:]
diameter4=np.append(templaal1,templaal2,axis=0)

templaal1=height[0:1673]
templaal2=height[2506:]
height4=np.append(templaal1,templaal2,axis=0)

templaal1=length[0:1673]
templaal2=length[2506:]
length4=np.append(templaal1,templaal2,axis=0)

templaal1=shell_weight[0:1673]
templaal2=shell_weight[2506:]
shell_weight4=np.append(templaal1,templaal2,axis=0)

templaal1=whole_weight[0:1673]
templaal2=whole_weight[2506:]
whole_weight4=np.append(templaal1,templaal2,axis=0)

templaal1=viscera_weight[0:1673]
templaal2=viscera_weight[2506:]
viscera_weight4=np.append(templaal1,templaal2,axis=0)

templaal1=shucked_weight[0:1673]
templaal2=shucked_weight[2506:]
shucked_weight4=np.append(templaal1,templaal2,axis=0)

templaal1=Y[0:1673]
templaal2=Y[2506:]
y4=np.append(templaal1,templaal2,axis=0)


x4 = np.c_[np.ones(diameter4.shape[0]),sex4,diameter4,height4,length4,shell_weight4,whole_weight4,viscera_weight4,shucked_weight4]
m = y4.size  
theta = np.random.rand(9)
prediction_list1, cost_list1, theta_list1 = gradient_descent(x2, y2, m, theta, alpha)
theta4 = theta_list1[-1]
# print('theta_gdfold1=',theta3)
predictions=x4.dot(theta4)
trainerrorfold4=y4-predictions
trainerrorgd4=np.square(trainerrorfold4)
sum1fromtrainerrorgd4=sum(trainerrorgd4)/len(trainerrorgd4)
rsmetrainerrorgd4=math.sqrt(sum1fromtrainerrorgd4)

rsmetrainingaverage.append(rsmetrainerrorgd4)
# normal equation
x_transpose = np.transpose(x4)   
transpose = x_transpose.dot(x4)  
temp_1 = np.linalg.inv(transpose) 
temp_2 = x_transpose.dot(y4) 
theta_normal4 = temp_1.dot(temp_2)
# print('theta_normalfold1=',theta_normal3)

predictions=x4.dot(theta_normal4)
trainerrornormal4=y4-predictions
trainerrornormal4=np.square(trainerrornormal4)
sum1fromtrainerrornormal4=sum(trainerrornormal4)/len(trainerrornormal4)
rsmetrainerrornormal4=math.sqrt(sum1fromtrainerrornormal4)
rsmetrainingaveragenormal.append(rsmetrainerrornormal4)



sexvalidate4=sex[1672:2506]
diametervalidate4=diameter[1672:2506]
heightvalidate4=height[1672:2506]
lengthvalidate4=length[1672:2506]
shell_weightvalidate4=shell_weight[1672:2506]
whole_weightvalidate4=whole_weight[1672:2506]
viscera_weightvalidate4=viscera_weight[1672:2506]
shucked_weightvalidate4=shucked_weight[1672:2506]
yvalidate4=Y[1672:2506]
xvalidate4= np.c_[np.ones(diametervalidate4.shape[0]),sexvalidate4,diametervalidate4,heightvalidate4,lengthvalidate4,shell_weightvalidate4,whole_weightvalidate4,viscera_weightvalidate4,shucked_weightvalidate4]

# print(y1)
# print(xvalidate1.dot(theta_normal))

predictionsfromgd=xvalidate4.dot(theta4)
differencegd=np.subtract(yvalidate4,predictionsfromgd)
errorgd4=np.square(differencegd)
sum1fromerrorgd4=sum(errorgd4)/len(errorgd4)
rsmeerrorgd4=math.sqrt(sum1fromerrorgd4)
rsmevalidationaverage.append(rsmeerrorgd4)

predictionsfromnormal=xvalidate4.dot(theta_normal4)
differencenormal=np.subtract(yvalidate4,predictionsfromnormal)
errornormal4=np.square(differencenormal)
sum1fromerrornormal4=sum(errornormal4)/len(errornormal4)
rsmeerrornormal4=math.sqrt(sum1fromerrornormal4)
rsmevalidationaveragenormal.append(rsmeerrornormal4)
# print(rsmeerrornormal4)



# ===========================================================

# fold5

alpha = 0.001

templaal1=sex[0:2506]
templaal2=sex[3340:]
sex5=np.append(templaal1,templaal2,axis=0)

templaal1=diameter[0:2506]
templaal2=diameter[3340:]
diameter5=np.append(templaal1,templaal2,axis=0)

templaal1=height[0:2506]
templaal2=height[3340:]
height5=np.append(templaal1,templaal2,axis=0)

templaal1=length[0:2506]
templaal2=length[3340:]
length5=np.append(templaal1,templaal2,axis=0)

templaal1=shell_weight[0:2506]
templaal2=shell_weight[3340:]
shell_weight5=np.append(templaal1,templaal2,axis=0)

templaal1=whole_weight[0:2506]
templaal2=whole_weight[3340:]
whole_weight5=np.append(templaal1,templaal2,axis=0)

templaal1=viscera_weight[0:2506]
templaal2=viscera_weight[3340:]
viscera_weight5=np.append(templaal1,templaal2,axis=0)

templaal1=shucked_weight[0:2506]
templaal2=shucked_weight[3340:]
shucked_weight5=np.append(templaal1,templaal2,axis=0)

templaal1=Y[0:2506]
templaal2=Y[3340:]
y5=np.append(templaal1,templaal2,axis=0)


x5 = np.c_[np.ones(diameter5.shape[0]),sex5,diameter5,height5,length5,shell_weight5,whole_weight5,viscera_weight5,shucked_weight5]
m = y5.size  
theta = np.random.rand(9)
prediction_list1, cost_list1, theta_list1 = gradient_descent(x2, y2, m, theta, alpha)
theta5 = theta_list1[-1]
# print('theta_gdfold1=',theta3)
predictions=x5.dot(theta5)
trainerrorfold5=y5-predictions
trainerrorgd5=np.square(trainerrorfold5)
sum1fromtrainerrorgd5=sum(trainerrorgd5)/len(trainerrorgd5)
rsmetrainerrorgd5=math.sqrt(sum1fromtrainerrorgd5)

rsmetrainingaverage.append(rsmetrainerrorgd5)
# normal equation
x_transpose = np.transpose(x5)   
transpose = x_transpose.dot(x5)  
temp_1 = np.linalg.inv(transpose) 
temp_2 = x_transpose.dot(y5) 
theta_normal5 = temp_1.dot(temp_2)
# print('theta_normalfold1=',theta_normal3)


predictions=x5.dot(theta_normal5)
trainerrornormal5=y5-predictions
trainerrornormal5=np.square(trainerrornormal5)
sum1fromtrainerrornormal5=sum(trainerrornormal5)/len(trainerrornormal5)
rsmetrainerrornormal5=math.sqrt(sum1fromtrainerrornormal5)
rsmetrainingaveragenormal.append(rsmetrainerrornormal5)





sexvalidate5=sex[2506:3340]
diametervalidate5=diameter[2506:3340]
heightvalidate5=height[2506:3340]
lengthvalidate5=length[2506:3340]
shell_weightvalidate5=shell_weight[2506:3340]
whole_weightvalidate5=whole_weight[2506:3340]
viscera_weightvalidate5=viscera_weight[2506:3340]
shucked_weightvalidate5=shucked_weight[2506:3340]
yvalidate5=Y[2506:3340]
xvalidate5= np.c_[np.ones(diametervalidate5.shape[0]),sexvalidate5,diametervalidate5,heightvalidate5,lengthvalidate5,shell_weightvalidate5,whole_weightvalidate5,viscera_weightvalidate5,shucked_weightvalidate5]

# print(y1)
# print(xvalidate1.dot(theta_normal))

predictionsfromgd=xvalidate5.dot(theta5)
differencegd=np.subtract(yvalidate5,predictionsfromgd)
errorgd5=np.square(differencegd)
sum1fromerrorgd5=sum(errorgd5)/len(errorgd5)
rsmeerrorgd5=math.sqrt(sum1fromerrorgd5)
rsmevalidationaverage.append(rsmeerrorgd5)

predictionsfromnormal=xvalidate5.dot(theta_normal5)
differencenormal=np.subtract(yvalidate5,predictionsfromnormal)
errornormal5=np.square(differencenormal)
sum1fromerrornormal5=sum(errornormal5)/len(errornormal5)
rsmeerrornormal5=math.sqrt(sum1fromerrornormal5)
rsmevalidationaveragenormal.append(rsmeerrornormal5)
# print(rsmeerrornormal5)


# ===========================================================
print('Training RMSE for GD= ',rsmetrainingaverage)
print('Validation RMSE for GD= ',rsmevalidationaverage)

print('Training RMSE for NormalEqn= ',rsmevalidationaveragenormal)
print('Validation RMSE for NormalEqn= ',rsmevalidationaveragenormal)

# x=[1,2,3,4,5]
# plt.plot(x, rsmetrainingaverage, color='blue',label='trainingfromgd')
# plt.plot(x, rsmevalidationaverage, color='g',label='rsmefromgd')
# plt.legend(loc='upper right')   
# plt.show()

x=[1,2,3,4,5]
plt.plot(x, rsmetrainingaveragenormal, color='blue',label='trainingfromNormal')
plt.plot(x, rsmevalidationaveragenormal, color='g',label='rsmefromNormal')
plt.legend(loc='upper right')   
plt.show()
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import warnings
# import math
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score,mean_squared_error
# from sklearn.feature_selection import SelectFromModel
# from sklearn.cluster import KMeans

# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score,mean_squared_error
# linear=LinearRegression()
# y = df['Rings']
# x = np.c_[diameter,height,length,shell_weight,whole_weight,viscera_weight,shucked_weight]
# linear.fit(x3,y3)
# pred = linear.predict(xvalidate3)
# print('from library= ',math.sqrt(mean_squared_error(yvalidate3, pred)))
# intercept = linear.intercept_
# Theta_0 = linear.coef_[0]
# Theta_1 = linear.coef_[1]
# print('yaba daba doodle')
# print(Theta_0)
# print(Theta_1)
# from sklearn import model_selection
# from sklearn.linear_model import LinearRegression
# array = df.values
# X = array[:,0:7]
# Y = array[:,7]
# # print(Y)
# num_folds = 10
# num_instances = len(X)
# loocv = model_selection.LeaveOneOut()
# model = LinearRegression()
# results = model_selection.cross_val_score(model, X, Y, cv=loocv)
# print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
# ===========================================================
