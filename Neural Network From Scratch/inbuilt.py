from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X / 255.
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

XTrain,Xvalidation=X_train[:50000],X_train[50000:]
YTrain,Yvalidation=y_train[:50000],y_train[50000:]

print("ReLu")
mlp = MLPClassifier(hidden_layer_sizes=(256,128,64),verbose=15, max_iter=15, alpha=1e-4,solver='sgd' ,warm_start=False,tol=1e-4, random_state=1,learning_rate_init=0.1,activation='relu')
mlp.fit(XTrain, YTrain)
p=mlp.loss_ 
print(p)
pred1=mlp.predict(Xvalidation)
print("Training set score ReLu: ", accuracy_score(Yvalidation,pred1))
pred2=mlp.predict(X_test)
print("Test set score ReLu: ",accuracy_score(y_test,pred2))

print("Linear")
mlp = MLPClassifier(hidden_layer_sizes=(256,128,64),verbose=15, max_iter=15, alpha=1e-4,solver='sgd', tol=1e-4, random_state=1,learning_rate_init=0.1,activation='identity')
mlp.fit(XTrain, YTrain)
pred1=mlp.predict(Xvalidation)
print("Training set score Linear: ", accuracy_score(Yvalidation,pred1))
pred2=mlp.predict(X_test)
print("Test set score Linear: ",accuracy_score(y_test,pred2))

print("Tanh")
mlp = MLPClassifier(hidden_layer_sizes=(256,128,64), verbose=15,max_iter=15, alpha=1e-4,solver='sgd', tol=1e-4, random_state=1,learning_rate_init=0.1,activation='tanh')
mlp.fit(XTrain, YTrain)
pred1=mlp.predict(Xvalidation)
print("Training set score Tanh: ", accuracy_score(Yvalidation,pred1))
pred2=mlp.predict(X_test)
print("Test set score Tanh: ",accuracy_score(y_test,pred2))

print("Sigmoid")
mlp = MLPClassifier(hidden_layer_sizes=(256,128,64), verbose=15,max_iter=15, alpha=1e-4,solver='sgd', tol=1e-4, random_state=1,learning_rate_init=0.1,activation='logistic')
mlp.fit(XTrain, YTrain)
pred1=mlp.predict(Xvalidation)
print("Training set score Sigmoid: ", accuracy_score(Yvalidation,pred1))
pred2=mlp.predict(X_test)
print("Test set score Sigmoid: ",accuracy_score(y_test,pred2))