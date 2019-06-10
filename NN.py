#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 10:32:21 2019

@author: Abdel Ghani Labassi
Inspired from Andrew NG's lectures on ML
"""
import numpy as np
import copy
from utils import form_matrix, accuracy, gen_dataset, sigmoid, dsigmoid

# An implementation of neural network learning algorithm using crossentropy
# cost function and gradient descent for its minimization. Gradient is computed
# according to the backpropagation algorithm
class NeuralNetworkCLF:
    
    
    def __init__(self, hiden_layers_sizes ):
    
        self.hiden_layers_sizes = hiden_layers_sizes 
        
        self.nlayers = len(hiden_layers_sizes) + 2 #io layers
    
    
      
    #X and Y are assumed to be list of list, where lines corresponds to instances of
    #the dataset.
    def fit(self,X,Y, learning_rate = 0.5, regularization_parameter = 0.0005, epsilon = 0.05, gradient_checking = False):
        
        # sizes[l] gives the number of neurons in layer l.
        sizes = [ len(X[0]) ] + list(self.hiden_layers_sizes) + [ len(Y[0]) ]
        
        
        #W[l][i,j] gives weigth of the arc (j,i), node i being in layer
        #number l, input layer being layer number 0.
        self.W = np.array( [  np.random.rand( int(sizes[l+1]), int(sizes[l] + 1)  ) 
                                                 for l in range( self.nlayers - 1 ) ] )
        
        J_calculator = self._get_J_calculator( X, Y, regularization_parameter )
        gradJ_caclulator = self._get_gradJ_calculator( X,Y, regularization_parameter )
        
        #Gradient descent
        while( True ):
            
            #Computing error
            error = np.abs( J_calculator( self.W  )  ) 
            print( "error:", error)
            
            #Stopping criterion
            if error < epsilon:
                break
            
            #Computing gradient
            gradJ = gradJ_caclulator( self.W )
            
            if gradient_checking :
                
                gradJ_approx = self._approx_gradJ( J_calculator, epsilon = 0.0005 )
                
                print( "gradient checking error:", 
                      max( [ np.amax( np.abs( gradJ_approx[l] - gradJ[l] ) ) 
                                               for l in range( len( self.W ) ) ] ) ) 
            
            #Perforiming a single gradient descent iteation
            for i in range( len( self.W ) ): 
                    self.W[i] = self.W[i] - learning_rate*gradJ[i]
                    
            
                    
      
    #Cost function : Evaluates the quality of the parameters depending on the 
    #training examples    ..
    def _get_J_calculator( self, X, Y, regularization_parameter ):
        
        def J_calculator( W ):

            dataset_size = len(X)
            
            #cost for a single coordinate of the output
            cost1D = lambda oj,yj: -( yj * np.log( oj ) + (1 - yj) * np.log( 1-oj ) )    

            fitting_term =0
            
            for i in range(dataset_size):
                
                output_i, target_i = self._output( X[i] ), Y[i]
                for j in range(len(Y[0])):              
                    fitting_term = fitting_term + cost1D( output_i[j], target_i[j] )
                    
            fitting_term = fitting_term / dataset_size 
            
            #We dont include biases in reg. term
            regularization_term = 0
            
            for l in range(len( W )):
                regularization_term =  np.sum( W[l][:,1:] )  
                
            regularization_term = regularization_parameter * regularization_term / (2*dataset_size)
            
            return fitting_term + regularization_term
        
        return J_calculator
    
    
    #Gives an approximaiton of gradient using the naive, ineficient method
    def _approx_gradJ( self, J_clc, epsilon = 0.0005 ):
        
        W = self.W
        res = np.array([ np.zeros( W[l].shape ) for l in range( len(W) ) ])

        for l in range( len( res )):
            for i in range( len( res[l] )):
                for j in range( len(res[l][0] )):
                    WR = copy.deepcopy( W[l] )
                    WR[i,j] = WR[i,j] + epsilon
                    WR = list( W[:l] ) + list( [WR] ) + list( W[l+1:] )
                    WL = copy.deepcopy( W[l] )
                    WL[i,j] = WL[i,j] - epsilon
                    WL =  list( W[:l] ) + list( [WL] ) + list( W[l+1:] )
                    
                    res[l][i,j] = (J_clc( WR ) - J_clc( WL )) /  epsilon
        
        return res

                        
               
    # Computes gradient of cost Function given dataset using backprop
    #return result in a list of matrix in same format as W for convenciency
    #i.e D[l][j][i] contains derivative for the weigth associated with 
    #edge ij in layer l.
    def _get_gradJ_calculator(self, X, Y, regularization_parameter ):
        
        def gradJ_calculator( W ):

            m = len(X)
            #acc is used to compute gradJ 
            acc = [ np.zeros(w.shape) for w in self.W ]
            
            #Computing acc
            for i in range(m):
                
                activations = self._forward_propagate( X[i], self.nlayers - 1 )
                
                all_dJdis = self._compute_dJdis( activations, Y[i] )
                
                #Upgrade acc
                for l in range(len(acc)) :
                    
                    dJdi = np.transpose( [all_dJdis[l+1]] )
                    
                    didw = [ activations[l] ]
                    
                    acc[l] = acc[l] + np.matmul( dJdi, didw )
        
            #Form gradient by normalizing acc
            acc_normalizer = lambda l: lambda i,j: acc[l][i,j] / len(X) + \
                                            bool(j) * regularization_parameter * self.W[l][i,j]
            
            
            #We store the partial derivative in a datastructure of exact same format as self.W
            
            return np.array( [ form_matrix( acc_normalizer(l), acc[l].shape ) 
                                             for l in range( len(acc) ) ] ) 
        
        return gradJ_calculator
        
    
    
    #Computes Djdi of each neurons n. DJdi is the derivative of J in respect 
    #of the inputs of each neuron n.
    #We define "input of a neuron" as the value what comes to it activation function,
    #We define "output of a neuron" as it activation.
    #We similarly define "input/output of a layer"
    def _compute_dJdis(self, activations, y):
        
        res = [0]*self.nlayers  #No error in layer 0, so res[0] 
        res[-1] = np.array( activations[-1][1:] - y )
        for i in range(self.nlayers - 2, 0, -1):
            
            #Derivative of error according to output of current layer, backpropaging
            #throw the arcs, ignoring bias unit because bias unit doens have entering arcs
            dJdo  = np.transpose( np.dot(np.transpose(self.W[i]), res[i+1] ))[1:]
            
            #Derivative of output according to inputs relative to the current layer.
            dodi = dsigmoid(np.dot(self.W[i-1], activations[i-1]))
            
            res[i] = dJdo * dodi
        
        return  res  
    
    
    
    #Returns the activations given the forward prop. of input x until the lth
    #layer, including the lth layer, and including inputs activation. We also 
    #include bias units 's output, wich is 1 for every layer.
    def _forward_propagate(self, x, l):
        
        ai,i = np.concatenate( ([1], x) ) , 0
        activations = [ ai ]
        
        #Forward propagation
        while i < l:
            
            ai = sigmoid( np.dot( self.W[i], ai) )
            ai =  np.concatenate( ([1], ai) )
            
            activations.append( ai )
            i = i + 1 
        
        return activations

            
    #Compute the output of the neural net given an input x
    def _output( self, x ):
        
        return list( self._forward_propagate( x, self.nlayers - 1  )[-1][1:] )
    
    
    #Gives an intepretation of what is displayed on output layer for each instances
    #of X
    def predict(self, X):
        
        res = [ self._output(x) for x in X ]
        bin_clf = len( res[0] )  == 1
        
        for i in range( len(res) ):
            
            if bin_clf:
                res[i][0] = 1 if res[i][0] > 0.5 else 0
            
            else:
                max_idx = res[i].index( max(res[i]) )
                
                for j in range( len(res[0] ) ):
                    if j == max_idx:
                        res[i][j] = 1
                    else:
                        res[i][j] = 0
                
        return res
                
    
        
        
        
nn = NeuralNetworkCLF( [2] )

X_train, X_test, y_train, y_test = gen_dataset( np.logical_xor, 4, sep = 0.8 )

X, y = X_train + X_test, y_train + y_test

nn.fit( X_train, y_train, epsilon = 0.05 , learning_rate = 0.15 , regularization_parameter = 0.0005 )

predicted = nn.predict( X )

#We try in all instances because dataset is small
print( accuracy( predicted, y ) )
