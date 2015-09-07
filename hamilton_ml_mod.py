from __future__ import print_function
from astropy.io import ascii
import matplotlib.pyplot as plt
from random import randint
import numpy as np
import pickle

from sklearn.preprocessing import Imputer,StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion , make_pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA,FastICA
from sklearn.metrics import make_scorer,mean_squared_error
from sklearn.linear_model import LogisticRegression,BayesianRidge,RANSACRegressor
from sklearn.neural_network import BernoulliRBM
from sklearn.ensemble import BaggingRegressor,RandomForestRegressor, AdaBoostRegressor,GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import FeedForwardNetwork,FullConnection,LinearLayer, SigmoidLayer, SoftmaxLayer, TanhLayer, GaussianLayer



#standardize data for imputation 
def standardize(data):
	size=data.shape
	for i in range(0,size[0]):
		try:
			for j in range(0,size[1]):
				if data[i][j]>90 or data[i][j]<-90:
					data[i][j]=-99
		except:
			if data[i]>90 or data[i]<-90:
					data[i]=-99
	return(data)
          
    

#Nueral Net Creation Function:
def create_net(n_features,layer_struct,function='Sigmoid'):#return(net) 

	#takes in a structure vector and creates a feed forward network
	#catch errors
	if min(layer_struct)<1:
		print('cannot create network')
		return 99.

	#initialize variables
	layers=[]
	connections=[]
	net = FeedForwardNetwork()

	#generate layers
	LI = LinearLayer(n_features)
	LO = LinearLayer(1)
	net.addInputModule(LI)
	net.addOutputModule(LO)
	for i in range(0,len(layer_struct)):
		if function=='Sigmoid':
			layers.append(SigmoidLayer(layer_struct[i]))
		elif function=='Softmax':
			layers.append(SoftmaxLayer(layer_struct[i]))
		elif function=='Linear':
			layers.append(LinearLayer(layer_struct[i]))
		elif function=='Gaussian':
			layers.append(GaussianLayer(layer_struct[i]))
		elif function=='Tanh':
			layers.append(TanhLayer(layer_struct[i]))
		else:
			print('Unknown function: Reverting to Sigmoid')
			layers.append(SigmoidLayer(layer_struct[i]))
		net.addModule(layers[i])

	#connect layers
	CI= FullConnection(LI, layers[0])
	CO = FullConnection(layers[-1], LO)
	net.addConnection(CI)
	net.addConnection(CO)
	for i in range(0,len(layer_struct)-1):
		connections.append(FullConnection(layers[i], layers[i+1]))
		net.addConnection(connections[i])

	net.sortModules()
	return net
    
#Train the network
def train_net(TrainDS,net,iterations): #return()
	trainer = BackpropTrainer(net, TrainDS)
	#trainer.trainUntilConvergence(dataset=TrainDS, maxEpochs=20, verbose=True)
	for i in range(0,iterations):
		trainer.train()
	return
	
#Evaluate the Network
def evaluate_net(TestDS,net): #return(results,MSE,eta,sig)  
	results=np.zeros(len(TestDS))
	for k in range(0,len(TestDS)):
		results[k] = net.activate(TestDS['input'][k])	
	return(results)

def metrics(X,Y): #return(MSE,eta,sig)
    #MSE calculation
    MSE = ((X-Y) ** 2).mean(axis=0)
    #Phot_z metrics
    delta_z=(X-Y)/(1+X)
    ntot=len(delta_z)
    nout=0.
    for i in range(0,ntot):
        if abs(delta_z[i])>.15:
            nout=nout+1
    eta=100.*nout/ntot
    sig=1.48*np.median(abs(delta_z))
    return(MSE,eta,sig)

def sigmaNMAD(X,Y): #return(MSE,eta,sig)
    delta_z=(X-Y)/(1+X)
    sig=1.48*np.median(abs(delta_z))
    return(sig)

def skl2pb(X,Y):
	DS = SupervisedDataSet(X.shape[1], 1)
	for i in range(0,len(X)):
		DS.addSample(X[i],Y[i])
	return(DS)
	
def csv_add(filename,array,add=True):
	if add:
		f=open(filename, 'a')
	else:
		f=open(filename, 'w')
	
	for j in range(0,len(array)-1):
		print(array[j],file=f,end=',')
	print(array[-1],file=f)
	
	f.close()
   

def subsample(data_list, n, axis=0):
	index_array = np.arange(data_list[0].shape[axis])
	np.random.shuffle(index_array)
	new_list=[]
	for X in data_list:
		sample=False
		if len(X.shape)>=axis+1 and X.shape[axis]>n:
			sample=True
		elif len(X.shape)<axis+1: 
			print('Dim error')
		
		if sample:
			if len(X.shape)==2 and axis==0:
				X = X[index_array[0:n],:]
			if len(X.shape)==2 and axis==1:
				X = X[:,index_array[0:n]]
			if len(X.shape)==1:		
				X = X[index_array[0:n]]	
		new_list.append(X)
	print('Data Subsampled')
	return new_list

def astro_plot(X,Y,title):	
	x2i = np.arange(0., max(X), 0.1)
	line2 = x2i
	sigup1 = x2i+.05*(1+x2i) 
	sigup2 = x2i+.15*(1+x2i) 
	sigdown1 = x2i-.05*(1+x2i) 
	sigdown2 = x2i-.15*(1+x2i) 

	(MSE,eta,sig)=metrics(X,Y)
	plt.scatter(X,Y,marker='.',s=20)
	plt.plot(x2i,line2,'r-',x2i,sigup1,'r--',x2i,sigdown1,'r--',x2i,sigup2,'r:',x2i,sigdown2,'r:',linewidth=1.5)
	plt.xlabel('$z_{spec}$',fontsize=14)
	plt.ylabel('$z_{phot}$',fontsize=14)
	plt.title(title)
	plt.annotate('$N_{tot} ='+str(len(X))+'$', xy=(.01, .9),xycoords='axes fraction',xytext=(0.03, 0.8), textcoords='axes fraction',fontsize=14)
	plt.annotate('$\eta ='+str(round(eta,2))+'\% $', xy=(.01, .2),xycoords='axes fraction',xytext=(0.03, 0.7), textcoords='axes fraction',fontsize=14)
	plt.annotate('$\sigma_{NMAD} ='+str(round(sig,3))+'$', xy=(.01, .2),xycoords='axes fraction',xytext=(0.03, 0.6), textcoords='axes fraction',fontsize=14)

def clf_iterator(X_train,Y_train,X_test,clf_list):
	res_list_test=[]
	res_list_train=[]
	for clf in clf_list:
		clf.fit(X_train, Y_train) 
		res_list_train.append(clf.predict(X_train))
		res_list_test.append(clf.predict(X_test))
	return(res_list_train,res_list_test)

class NueralNet(BaseEstimator, ClassifierMixin):
	def __init__(self,layer_struct=[20,20],iterations=200,function='Sigmoid'):
		self.layer_struct = layer_struct
		self.iterations = iterations
		self.function = function

	def fit(self, X, Y):
		self.n_features = X.shape[1]
		self.net = create_net(self.n_features,self.layer_struct,self.function)
		TrainDS = skl2pb(X,Y)
		train_net(TrainDS,self.net,self.iterations)
		return self
		
	def score(self, X, Y):
		results = self.predict(X)
		score = mean_squared_error(results,Y)
		return score

	def predict(self,X):
		results=np.zeros(len(X))
		for k in range(0,len(X)):
			results[k] = self.net.activate(X[k])
		return results


def dictlist_add(dict_list,dict_add):
	for i in dict_list:		
		for j in dict_add:
			i[j]=dict_add[j]
	return dict_list

def addstring(dict_list, string):
	for dict in dict_list:
		old_keys=[]
		new_keys=[]
		for k in dict:
			old_keys.append(k)
			new_keys.append(string+k)
		for old_key,new_key in zip(old_keys,new_keys): 
			dict[new_key] = dict.pop(old_key)
	return dict_list

def fit_list(reg_list,X_train,Y_train,X_test,Y_test):
	Y_actual_train_l = []
	Y_actual_test_l  = []
	Y_pred_train_l   = []
	Y_pred_test_l    = []

	for reg in reg_list:
		reg.fit(X_train,Y_train)
		Y_actual_train_l.append(Y_train)
		Y_actual_test_l.append(Y_test)
		Y_pred_train_l.append(reg.predict(X_train))
		Y_pred_test_l.append(reg.predict(X_test))
	
	return(Y_actual_train_l,Y_actual_test_l,Y_pred_train_l,Y_pred_test_l)

def bootstrap_fit(reg,X,Y,test_size=.4,n_iters=5):
	Y_actual_train= np.array([])
	Y_actual_test = np.array([])
	Y_pred_train  = np.array([])
	Y_pred_test   = np.array([])
	
	for i in range(n_iters):
		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
		
		#print(X_train.shape,Y_train.shape,reg,i)
		reg.fit(X_train,Y_train)
		try:
			Y_pred_train=np.concatenate((Y_pred_train,reg.predict(X_train)))
			Y_pred_test=np.concatenate((Y_pred_test,reg.predict(X_test)))
		except:
			Y_pred_train=np.concatenate((Y_pred_train,reg.predict(X_train)[:,0]))
			Y_pred_test=np.concatenate((Y_pred_test,reg.predict(X_test)[:,0]))
		
		Y_actual_train=np.concatenate((Y_actual_train,Y_train))
		Y_actual_test=np.concatenate((Y_actual_test,Y_test))
	return(Y_actual_train,Y_actual_test,Y_pred_train,Y_pred_test)	
	
def bootstrap_fit_list(reg_list,X,Y,test_size=.4,n_iters=5):
	Y_actual_train_l = []
	Y_actual_test_l  = []
	Y_pred_test_l    = []
	Y_pred_train_l   = []
	
	for reg in reg_list:
		(Y_actual_train,Y_actual_test,Y_pred_train,Y_pred_test)=bootstrap_fit(reg,X,Y,test_size,n_iters)
		Y_actual_train_l.append(Y_actual_train)
		Y_actual_test_l.append(Y_actual_test)
		Y_pred_train_l.append(Y_pred_train)
		Y_pred_test_l.append(Y_pred_test)
		
	return(Y_actual_train_l,Y_actual_test_l,Y_pred_test_l,Y_pred_train_l)

def pickler(object,file_name):
	file = open(file_name, "w")
	pickle.dump(object,file)
	file.close()
	
def depickler(file_name):
	file = open(file_name, "r")
	object=pickle.load(file)
	file.close()
	return object

def parser(data_file,feature_file,response_var,parametric_col='None'):
	with open(feature_file) as f:
		features = f.readlines()
	features=map(str.strip, features)

	all_data = ascii.read(data_file,fill_values=[("", "-99"),("null","-99"),("--","-99"),("999999","-99")])
	feature_data=all_data[features]
	X = np.array([feature_data[c] for c in feature_data.columns ])
	X = np.transpose(X)
	Y = np.array(all_data[response_var])

	if parametric_col!='None': 
		P = np.array(all_data[parametric_col])
		P = np.transpose(P)    
		return (X,Y,P)
	else:
		P = []
		return(X,Y)