#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 10:32:21 2019

@author: Abdel Ghani Labassi

"""
import numpy as np
import copy
from utils import form_mat, accuracy, gen_dataset, sigmoid, dsigmoid

#An implementation of a fully connected neural network learning algorithm using crossentropy 
#cost function and gradient descent for its minimization.
class FullyConnectedNN:
    
    
    def __init__(self, hiden_layers_sizes):
        self.hiden_layers_sizes = hiden_layers_sizes 
        self.nlayers = len(hiden_layers_sizes) + 2 #io layers
    
   
    def fit(self,X, Y, lr=0.5, rp=0.0005, epsilon=0.05, check_gradient=False):
        
        sizes = [ len(X[0]) ] + list(self.hiden_layers_sizes) + [ len(Y[0]) ]
        
        
        #W[l][i,j] gives weigth of the arc (j,i), node i being in layer
        #number l, input layer being layer number 0.
        self.W = [ np.random.rand(int(sizes[l+1]), int(sizes[l] + 1)) for l in range( self.nlayers - 1 ) ] 
        
        J_calculator = self._get_J_calculator(X, Y, rp)
        gradJ_caclulator = self._get_gradJ_calculator(X, Y, rp)
        
        #Performing iterations of gradient descent
        while(True):
            
            #Computing error of current nn
            error = np.abs(J_calculator(self.W ))
            print( "error:", error)
            if error < epsilon:
                break
            
            #Updating weigths
            gradJ = gradJ_caclulator(self.W)
            
            if check_gradient :
                gradJ_approx = self._approx_gradJ(J_calculator, epsilon=0.0005)
                max_err = max( [ np.amax( np.abs( gradJ_approx[l] - gradJ[l] ) ) for l in range( len( self.W ) ) ] )
                print( "gradient checking error:", max_err) 
            
            for l in range(len( self.W )): 
                    self.W[l] = self.W[l] - lr*gradJ[l]
                    
            
                    
      
    #Cost function : Evaluates the quality of the parameters depending on the training examples
    def _get_J_calculator(self, X, Y, rp):
        
        dataset_size = len(X)
        
        def J_calculator(W):
            
            #cost for a single coordinate of the output
            cost1D = lambda oj, yj: -( yj * np.log(oj) + (1-yj) * np.log(1-oj) )    

            fit_term =0
            for i in range(dataset_size):
                output_i, target_i = self._output(X[i]), Y[i]
                for j in range(len(Y[0])):   
                    fit_term = fit_term + cost1D(output_i[j], target_i[j])
            
            reg_term = 0
            for l in range(len(W)):  
                reg_term = reg_term + np.sum(W[l][:,1:]) #We dont include biases in reg. term 
            
            #Normalizing terms
            fit_term = fit_term / dataset_size 
            reg_term = rp * reg_term / (2*dataset_size)
            
            return fit_term + reg_term
        
        return J_calculator
    
    
    #Gives an approximaiton of gradient using the naive, ineficient method
    def _approx_gradJ(self, J_clc, epsilon = 0.0005):
        
        W = self.W
        res = [ np.zeros( W[l].shape ) for l in range( len(W) ) ]

        for l in range(len(res)):
            for i in range(len(res[l])):
                for j in range(len(res[l][0])):            
                    WR = copy.deepcopy( W[l] )
                    WR[i,j] = WR[i,j] + epsilon
                    WR = list( W[:l] ) + list( [WR] ) + list( W[l+1:] )
                    WL = copy.deepcopy( W[l] )
                    WL[i,j] = WL[i,j] - epsilon
                    WL =  list( W[:l] ) + list( [WL] ) + list( W[l+1:] )
                    
                    res[l][i,j] = (J_clc( WR ) - J_clc( WL )) /  epsilon
        
        return res

                        
               
    # Computes gradient of cost Function given dataset using backprop
    #return result in a list of matrix in same format as W,
    #i.e D[l][j][i] contains derivative for the weigth associated with 
    #edge ij in layer l.
    def _get_gradJ_calculator(self, X, Y, rp):
        
        dataset_size = len(X)
        
        def gradJ_calculator(W):
            
            dJdws_acc = [ np.zeros(w.shape) for w in self.W ] #used to compute gradJ 
            
            for i in range(dataset_size):
                acts = self._forward_propagate(X[i], self.nlayers-1)
                dJdis = self._compute_dJdis(acts, Y[i])
                
                #Upgrade acc
                for l in range(len(W)) :
                    dJdi = np.transpose([dJdis[l+1]])
                    didw = [acts[l]]
                    dJdws_acc[l] = dJdws_acc[l] + np.matmul(dJdi, didw)
        
            #Form gradient by normalizing acc
            acc_normalizer = lambda l: lambda i,j: dJdws_acc[l][i,j]/dataset_size + bool(j)*rp*self.W[l][i,j]
            
            
            #We store the partial derivative in a datastructure of exact same format as self.W
            
            return [ form_mat(acc_normalizer(l), dJdws_acc[l].shape) for l in range(len(W)) ] 
        
        return gradJ_calculator
        
    
    
    #Computes Djdi of each neurons n. DJdi is the derivative of J in respect of the inputs of each neuron n.
    #We define "input of a neuron" as the things that comes to it activation function,
    #We define "output of a neuron" as it activation.
    #We similarly define "input/output of a layer"
    def _compute_dJdis(self, acts, y):
        
        dJdis = [0]*self.nlayers  #No error in layer 0, so res[0] = 0
        dJdis[-1] = acts[-1][1:] - y
        
        for l in range(self.nlayers - 2, 0, -1):
            
            #Derivative of error according to output of current layer, computed using
            #dJdi of the next layer, by backpropaging through weighted arcs, ignoring bias
            #cause bias units dont have entering arcs.
            dJdo  = np.transpose(np.dot(np.transpose(self.W[l]), dJdis[l+1]))[1:]
            
            #Derivative of output according to inputs relative to the current layer.
            dodi = dsigmoid(np.dot(self.W[l-1], acts[l-1]))
            
            dJdis[l] = dJdo * dodi
        
        return  dJdis 
    
    
    
    #Returns the acts given the forward prop. of input x until the lth layer, including the lth layer,
    #and including inputs activation. We also include bias units 's output, wich is 1 for every layer.
    def _forward_propagate(self, x, l):
        
        ai,i = np.concatenate(([1], x)), 0
        acts = [ai]
        
        #Forward propagation
        while i < l:
            ai = sigmoid(np.dot( self.W[i], ai))
            ai =  np.concatenate(([1], ai))
            acts.append(ai)
            i = i + 1 
        
        return acts

            
    #Compute the output given by the neural net given an input x
    def _output(self, x):
        return  self._forward_propagate(x, self.nlayers-1)[-1][1:]
    
    
    #Gives an intepretation of what is displayed on output layer for each instances of X
    def predict(self, X):
        
        res = [ self._output(x) for x in X ]
        output_dim = len(res[0])
        bin_clf = output_dim  == 1
        
        for i in range(len(X)):
            if bin_clf:
                res[i][0] = 1 if res[i][0] > 0.5 else 0
            
            #one-hot encoding
            else:
                max_idx = res[i].index(max(res[i]))
                
                for j in range(output_dim):
                    if j == max_idx:
                        res[i][j] = 1
                    else:
                        res[i][j] = 0
                
        return res
                
    
        
        
        
nn = FullyConnectedNN([2])

X_train, X_test, y_train, y_test = gen_dataset(np.logical_xor, 4, sep=0.8)

X, y = X_train + X_test, y_train + y_test

#Problem : vanishing gradient
nn.fit(X_train, y_train, epsilon=0.24 , lr=0.15 , rp=0.0005, check_gradient=False)

predicted = nn.predict(X_test)

#We try in all instances because dataset is small
print(accuracy( predicted, y_test))
