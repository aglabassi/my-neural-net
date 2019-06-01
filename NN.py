#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 10:32:21 2019

@author: Abdel Ghani Labassi
Inspired from Andrew NG's lectures on ML
"""
import numpy as np
from utils import *

#An implementation of neural network learning algorithm using crossentropy cost
#function and gradient descent for its minimization. Gradient is computed according
#to the backpropagation algorithm.
class NeuralNetworkCLF:
  
    def __init__(self, hiden_layers_sizes ):
        
        self.hiden_layers_sizes = hiden_layers_sizes 
        
        self.nlayers = len( hiden_layers_sizes) + 2 #io layers
      
    
    
      
    #X and Y are assumed to be list of list, where lines corresponds to instances of
    #the dataset.
    def fit(self,X,Y, learning_rate = 0.5, regularization_parameter = 0.0005, epsilon = 0.05):
        
        # sizes[l] gives the number of neurons in layer l.
        sizes = [ len(X[0]) ] + list( self.hiden_layers_sizes ) + [ len(Y[0]) ]
        
        
        #W[l][i,j] gives weigth of the arc (j,i), node i being in layer number l, input layer being layer number 0.
        self.W = [  np.random.rand( int(sizes[l+1]), int(sizes[l] + 1)  ) 
                    for l in range( self.nlayers - 1 ) ]  #biases comming out of layer l are in W[l][:, 0 ]
        
        
        while( True ):
            
            error = np.abs( self._J( X, Y, regularization_parameter ) ) 
            print( error )
            
            if error < epsilon:
                break
            
            gradJ = self._gradJ( X, Y, regularization_parameter )
            
            #Gradient descent. Upgrade W layer by layer
            for i in range( len( self.W ) ): 
                    self.W[i] = self.W[i] - learning_rate * gradJ[i]
                    
            
                    
      
    #Cost function : Evaluates the quality of the parameters depending on the 
    #training examples    
    def _J( self, X, Y, regularization_parameter ):
        
        dataset_size = len( X )
    
            
        #cost for a single coordinate of the output
        cost1D = lambda oj, yj: -( yj * np.log( oj ) + (1 - yj) * np.log( 1-oj ) )    
        
        
        #Computing fitting term
        fitting_term = 0
        for i in range(dataset_size):    
            output_i, target_i = self._output( X[i] ), Y[i]
            for j in range( len( Y[0] ) ):
                fitting_term = fitting_term + cost1D( output_i[j], target_i[j] )
        fitting_term = fitting_term / dataset_size 
       
        
        #Computing regularization term
        regularization_term = 0
        for l in range(len(self.W)):  
            regularization_term =  np.sum( self.W[l][:,1:] )  #We dont include biases in reg. term
        regularization_term = regularization_parameter * regularization_term / (2*dataset_size)
        
        
        return fitting_term + regularization_term

                        
               
    # Computes gradient of cost Function given dataset using backprop
    #return result in a list of matrix in same format as W for convenciency
    #i.e D[l][j][i] contains derivative for the weigth associated with 
    #edge ij in layer l.
    def _gradJ(self,X,Y,regularization_parameter ):
        
        m = len(X)
        acc = [ np.zeros( w.shape ) for w in self.W ] #acc is used to compute gradJ 
        for i in range( m ):
            
            acts = self._forward_propagate( X[i], self.nlayers - 1 )
            
            all_dJdis = self._compute_DJdis( acts, Y[i] )
            
            #Upgrade acc
            for l in range( len( acc ) ) :
                
                dJdi = np.transpose( [all_dJdis[l+1]] )
                
                didw = [ acts[l] ]
                
                acc[l] = acc[l] + np.matmul( dJdi, didw )
    
        #Form gradient by normalizing acc
      
        acc_normalyzer = lambda l: lambda i,j: acc[l][i,j] / len(X) + \
                                        bool(j) * regularization_parameter * self.W[l][i,j]
        
        
        #We store the partial derivative in a datastructure of exact same format as self.W   
        gradJ = [ form_matrix( acc_normalyzer(l), acc[l].shape ) for l in range( len(acc) ) ] 
        
        return gradJ
    
    
    
    #Computes DJdi of each neurons n. DJdi is the derivative of J in respect 
    #of the inputs of each neuron n, stocked a natural list-of-list format.
    #We define "input of a neuron" as the thing that comes to it activation function, i.e wx + b,
    #We define "output of a neuron" as its activation.
    #We similarly define "input/output of a layer"
    def _compute_DJdis(self, activations, y):
        
        res = [0]*self.nlayers  #Neurons on layer 0 dont have inputs, so we keep res[0] = 0.
        res[-1] = np.array( activations[-1][1:] - y ) 
        
        for i in range(self.nlayers - 2, 0, -1):
            
            #Derivative of error according to output of current layer, backpropaging
            #throw the arcs, ignoring bias unit because bias unit doens have entering arcs
            dJdo  = np.transpose( np.dot(np.transpose(self.W[i]), res[i+1] ))[1:]
            
            #Derivative of output according to inputs relative to the current layer.
            dodi = dsigmoid( np.dot( self.W[i-1],  activations[i-1] )  )
            
            res[i] = dJdo * dodi
        
        return  res  
    
    
    
    #returns the activations given the forward prop. of input x until the lth layer, 
    #including the lth layer, and including inputs activation. We also 
    #include bias units 's output, wich is 1 for every layer.
    def _forward_propagate(self, x, l):
        
        ai,i = np.concatenate( ([1], x) ) , 0
        activations = [ ai ]
        
        #Forward propagation
        while i < l:
            
            ai = sigmoid( np.dot(self.W[i], ai) )
            ai =  np.concatenate( ([1], ai) )
            
            activations.append( ai )
            i = i + 1 
        
        return activations

            
    #Compute the output of the neural net given an input x
    def _output(self,x):
        return list( self._forward_propagate( x, self.nlayers - 1  )[-1][1:] )
    
    
    #Gives an intepretation of what is displayed on output layer for each instances of X
    def predict(self, X):
        
        res = [ self._output(x) for x in X ]
        
        bin_clf = len( res[0] )  == 1
        
        for i in range( len( res ) ):
            
            if bin_clf:
                res[i][0] = 1 if res[i][0] > 0.5 else 0
            
            else:
                max_idx = res[i].index( max(res[i]) ) 
                
                for j in range( len( res[0] ) ):
                    if j == max_idx:
                        res[i][j] = 1
                    else:
                        res[i][j] = 0
                
        return res
                
    
        
        
        
nn = NeuralNetworkCLF( [2 ])

X_train, X_test, y_train, y_test = gen_dataset_alrd_splitted( np.logical_xor, 2, sep = 0.8 )

X, y = X_train + X_test, y_train + y_test

nn.fit( X_train, y_train, epsilon = 0.05 , learning_rate = 0.15 , regularization_parameter = 0.0005 )

#We predict on all instances because dataset is small
predicted = nn.predict( X_train + X_test )

#We try in all instances because dataset is small
print( np.mean( predicted == (y_train + y_test) ) )
