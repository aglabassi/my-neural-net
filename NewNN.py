# -*- coding: utf-8 -*-
"""
Spyder Editor
Created 30 january 2021
This is a temporary script file.
"""

import numpy as np
from scipy.special import softmax


class NN():
    
    def relu(x):
        return np.vectorize(max)(x,0)
    
    def I(x): 
        return x
    
    
    #x is minibatch*output
    def softmax(x):
        
        return softmax(x,axis=1)

    
    
    def __init__(self, input_dim, hidden_dims, output_dim):
        
        self.dims = dims = [input_dim] + hidden_dims + [output_dim]

        self.n_layers = len(dims) - 1
        
        #init weights
        self.Ws = [ np.random.rand(dims[k], dims[k+1])  for k in range(self.n_layers) ]
        self.bs = [ np.random.rand(dims[k+1]) for k in range(self.n_layers) ]
        
        #init gradient
        self.dWs = [ np.zeros((dims[k], dims[k+1]))  for k in range(self.n_layers) ]
        self.dbs = [ np.zeros((dims[k+1])) for k in range(self.n_layers) ]
    
        self.activations = [ NN.relu for k in range(self.n_layers -1) ] + [ NN.softmax ]
        self.dactivations = [ NN.I ]*self.n_layers 
        
        
    #x minibatch
    def forward(self, x):
        
        self.hs = [x]
        self.ass = [x]
        
        #layer is define as w,b
        for k in range(0,self.n_layers):
            a = np.dot(x,self.Ws[k])+ self.bs[k] 
            self.ass.append(a)
            
            x = self.activations[k]( a )
            self.hs.append(x)
            
            
        return x
    
    def clear_gradient(self):
        dims = self.dims
        
        self.dWs = [ np.zeros((dims[k], dims[k+1]))  for k in range(self.n_layers) ]
        self.dbs = [ np.zeros((dims[k+1])) for k in range(self.n_layers) ]
        
    
    def step(self, learning_rate=0.01):
    
        for k in range(self.n_layers):
            
            self.Ws[k] = self.Ws[k] - learning_rate * self.dWs[k]
            self.bs[k] = self.bs[k] - learning_rate * self.dbs[k]
            
        
    
    
    
    def backprop(self, y):
        hs = self.hs
        ass = self.ass
        
        y_hat = hs[-1]
        
        assert(y.shape == y_hat.shape)
        
        dA = - (y_hat - y) #softmax
          
        #hs[k] gives the hypothesis at layer k, hs[0] = x, and h[nlayers] = f(x)
        for k in range(self.n_layers-1, -1 ,-1):
 
            self.dWs[k] = np.matmul(np.transpose(hs[k]), dA)/y.shape[0]
            self.dbs[k] = np.matmul(np.ones((1,y.shape[0])), dA)/y.shape[0]
            #print(self.dWs)  
            
            dH = np.matmul(dA, np.transpose(self.Ws[k]))
            dA = np.multiply(dH, self.dactivations[k](ass[k]))
            
                
            
        
            
        
        
        

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

model = NN(4,[2], 3)
data = load_iris()
target= to_categorical(data.target,3)
                 
X_train, X_test, y_train, y_test = train_test_split(data.data, target,train_size=0.2)
#train model:
    ##FIX EXPLODING GRADIEND
for i,x in enumerate(X_train):
    y = np.expand_dims(y_train[i], 0)
    x = np.expand_dims(x, 0)
    
    model.forward(x)
    model.backprop(y)
    model.step()
    model.clear_gradient()
    
    
print(np.mean(y_train == model.forward(X_train)))
    




        
        
        
        
        
