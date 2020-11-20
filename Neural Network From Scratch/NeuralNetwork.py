import numpy as np
import pickle
import activationfunctions
import random
from math import log10
import matplotlib.pyplot as plt
import os
# ============================================================
def plot(epochs,error,functionname):
	plt.plot(epochs,error)
	plt.xlabel('Epochs')
	plt.ylabel('Training Error')
	plt.title(functionname)
	plt.show()
class Network(object):
	def __init__(self,sizes,learning_rate,mini_batch_size,activation_fn):
		
		# assign values
		self.sizes = sizes
		self.mini_batch_size = mini_batch_size
		self.lr = learning_rate
		self.activation_fn = getattr(activationfunctions, activation_fn)
		self.activation_fn_prime = getattr(activationfunctions, f"{activation_fn}_prime")
		self.activationfunctionname=activation_fn
		# setup the neuralnet
		self.num_layers = len(sizes)
		self.weightflag=0
		#Input does not have any weight and no sense appending bias before it  therefore 0s 
		if activation_fn!="Linear":
			self.weightflag+=1
			# print('Size1= ',len(sizes[:]))
			# print('Size2= ',len(sizes[:-1]))
			self.weights =[np.array([0])]+[(np.random.randn(y, x)/np.sqrt(x)) for y, x in zip(sizes[1:], sizes[:-1])]
			
		else:
			self.weightflag+=1
			self.weights = [np.array([0])] + [(np.random.randn(y, x)/(np.sqrt(x))*100) for y, x in zip(sizes[1:], sizes[:-1])]
		
		self.trainingerror=[]
		self.validationerror=[]
		self.fitflag=0
		inputbiaspadding=[np.array([0])]
		self.biases =  inputbiaspadding+ [np.random.randn(y, 1) for y in sizes[1:]]
		self.backwardflag=0
		# this basically stores the output from the input layer and then updates in subsequent passes
		self._activations = [np.zeros(bias.shape) for bias in self.biases]
		#  transform is basically y=wx+b. This is where the actual backpropagation does changes 
		self.forwardflag=0
		self.transform = [np.zeros(bias.shape) for bias in self.biases]
		
	def forwardpass(self, x):
		self._activations[0] = x
		i=1
		while i < self.num_layers:
			# the classical wx
			dotproduct=self.weights[i].dot(self._activations[i - 1])
			bias=self.biases[i]
			last=self.num_layers - 1
			# calculating y=wx+b
			self.transform[i] = dotproduct + bias
			if i != last:
				self.forwardflag=0
				self._activations[i] = self.activation_fn(self.transform[i])	
			else:
				self.forwardflag=0
				self._activations[i] = activationfunctions.softmax(self.transform[i])
			i+=1

	def fit(self, training_data, validation_data, epochs,regularization):
		for epoch in range(epochs):
			random.shuffle(training_data)
			limit=self.mini_batch_size
			mini_batches=[training_data[k:k + limit] for k in range(0, len(training_data), limit)]
			ratio=self.lr/self.mini_batch_size
			for j in mini_batches:
				
				
				# layer-by-layer lists of numpy arrays, similar to self.biases and self.weights
				weight = [np.zeros(weight.shape) for weight in self.weights]
				self.fitflag=1
				bias = [np.zeros(bias.shape) for bias in self.biases]
				
				for x, y in j:
					self.forwardflag=1
					self.forwardpass(x)
					delta_bias, delta_weight = self.back_prop(x, y)

					bias = [b + db for b, db in zip(bias, delta_bias)]
					self.flag=0
					weight = [w + dw for w, dw in zip(weight, delta_weight)]
				self.weights = [w - (ratio) * dw  for w, dw in zip(self.weights, weight)]
				self.biases = [b - (ratio) * db for b, db in zip(self.biases, bias)]
			
			if self.activationfunctionname!="Linear":
				funcname=self.activationfunctionname
				training_error=self.training_error(training_data,funcname)
				validationerror=self.training_error(validation_data,funcname)
				self.trainingerror.append(training_error*ratio)
				self.validationerror.append(validationerror*ratio)
				print("Epoch ", epoch + 1,end=" ") 
				print("Error ",training_error*ratio)
			else:
				print("Epoch ", epoch + 1,end=" ") 
				print("Error from Linear Activation Not Calculatable")

			accuracy = self.validate(validation_data) / 100.0
			print("Epoch ", epoch + 1,end=" ") 
			print("Accuracy ",accuracy)
	
	def validate(self, validation_data):
		validation_results = [(self.predict(x) == y) for x, y in validation_data]
		return sum(result for result in validation_results)
	def back_prop(self, x, y):
		error = (self._activations[-1] - y)
		bias = [np.zeros(bias.shape) for bias in self.biases]
		self.backwardflag=1
		weight = [np.zeros(weight.shape) for weight in self.weights]
		bias[-1] = error
		weight[-1] = error.dot(self._activations[-2].transpose())
		# backprop beigns. Start with the second last layer all the way to input
		l=self.num_layers - 2
		while l >=0:
			transposedweights=self.weights[l + 1].transpose()
			dotproducterror=transposedweights.dot(error)
			antiderivativeoftransform=self.activation_fn_prime(self.transform[l])
			error = np.multiply(dotproducterror,antiderivativeoftransform)
			bias[l] = error
			weight[l] = error.dot(self._activations[l - 1].transpose())
			l-=1
		return bias, weight
	def saveparameters(self):
		filename=self.activationfunctionname+'.npz'
		np.savez_compressed(file=filename,weights=self.weights,biases=self.biases,mini_batch_size=self.mini_batch_size,lr=self.lr)
	def training_error(self,data,funcname):
		if funcname!="Linear":
			differences = [(self.predict(x) == y) for x, y in data]
			squarederror=(sum(diff for diff in differences))
			meanerror=(squarederror)/(2*len(data))
			rmseerror=np.sqrt(meanerror)
			return np.linalg.norm(rmseerror)
	def predict(self, x):
		self.forwardpass(x)
		return np.argmax(self._activations[-1])
# ===========================================================================================================
input1=int(input("Enter the Number of Hidden Layers: "))
input2=list(map(int,input("Enter Number of Nodes in Hidden Layer separated by space ").split()))
temp1=[784]
temp1.extend(input2)
temp1.append(10)
# print(temp1)
with open('train.txt','rb') as fp:
	train=pickle.load(fp)
with open('test.txt','rb') as fp:
	test=pickle.load(fp)
with open('val.txt','rb') as fp:
	val=pickle.load(fp)
while True:
	print("Choose activation function for the hidden Layers")
	print("1. Linear 2.Sigmoid 3.Tanh 4.ReLu")
	input3=int(input())

	if input3==1:
		input11=int(input("Enter Epochs "))
		nn = Network(temp1, 0.1, 16,"Linear")
		nn.fit(train, val, input11,0.1)
		accuracy = nn.validate(test) / 100.0
		print("Test Accuracy: ",str(accuracy)+"%")
		print("Training Error vs Epoch curve is not possible")
		nn.saveparameters()
	elif input3==2:
		epochs=list(range(15))
		# error=[0.004661762676284583,0.00450871557830452,0.004494180264519883,0.004444316946955967,0.00441570351133316,0.004425247168238176,0.0044165880496147706,0.004398685883306513,0.004404597589248651,0.0044020148085848155,0.004435298749802543,0.0044152209417548,0.004414066648228139,0.00441468116213797488,0.004419196406022253]
		# plot(epochs,error,'Sigmoid')
		input11=int(input("Enter Epochs "))
		nn = Network(temp1, 0.1, 16,"Sigmoid")
		nn.fit(train, val, input11,0.1)
		accuracy = nn.validate(test) / 100.0
		print("Test Accuracy: ",str(accuracy)+"%")
		nn.saveparameters()
		
	elif input3==3:
		epochs=list(range(15))
		# error=[0.000118479,0.000117523,0.000117070,0.000116542,0.000116242,0.000116221,0.000116827,0.000116794,0.000115949,0.000115689,0.000115043,0.000115027,0.000115027,0.000115027,0.000115027]
		
		# plot(epochs,error,'Tanh')
		input11=int(input("Enter Epochs "))
		nn = Network(temp1, 0.1, 16,"Tanh")

		nn.fit(train, val, input11,0.1)
		print(nn.trainingerror)
		print(nn.validationerror)
		accuracy = nn.validate(test) / 100.0
		print("Test Accuracy: ",str(accuracy)+"%")
		nn.saveparameters()
	
	elif input3==4:
		epochs=list(range(15))
		# error=[0.00442723284061276,0.004427303855557136,0.0044203011775217314,0.004423745604779824,0.004422314808584815,0.004422333235973969,0.004423452669763584,0.004418002942506942,0.004416543826908095,0.00441678945335226765,0.0044168600230321089,0.0044167693473408947,0.00441454093663942,0.004413525531179853,0.004413524531179853]
		# plot(epochs,error,'ReLu')
		input11=int(input("Enter Epochs "))
		nn = Network(temp1, 0.1, 16,"ReLu")
		nn.fit(train, val, input11,0.1)
		accuracy = nn.validate(test) / 100.0
		print("Test Accuracy: ",str(accuracy)+"%")
		nn.saveparameters()
	else:
		print('Sorry choose correctly next time')
		break