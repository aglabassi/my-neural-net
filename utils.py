#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:57:20 2019
Contains various useful functions

@author: Abdel
"""
import numpy as np

#Form a matrix M where M[i,j] == fc(i,j). 
#(fc is a 2-parameter function)
def form_matrix(fc, shape):
    
    assert(len(shape) == 2)
    
    res = np.zeros(shape)
    
    for i in range(shape[0]):      
        for j in range(shape[1]):
            res[i,j] = fc(i,j)
            
    return res

#Assert fc is logical binary fonction. Generate a training/test set according to
# fc and sep    
def gen_dataset(fc, input_dim, sep=0.8):
    
    X =  [ [ int(s) for s in np.binary_repr(i,input_dim) ] for i in range(2**input_dim)  ]
    
    Xpos,Xneg = [],[]
    ypos, yneg = [], []
    
    for x in X:
        y = fc.reduce(x)
        if y:
            ypos.append([y])
            Xpos.append(x)
            
        else:
            yneg.append([y])
            Xneg.append(x)
            
    tpos, tneg = int(np.around(sep*len(Xpos))), int(np.around(sep*len(Xneg)))
    X_train, y_train = Xpos[:tpos] +  Xneg[:tneg], ypos[:tpos] + yneg[:tneg]
    X_test, y_test = Xpos[tpos:] + Xneg[tneg:], ypos[tpos:] + yneg[tneg:]
    
    return X_train, X_test, y_train, y_test
    
def dsigmoid(z):
    
    return sigmoid(z)*(1-sigmoid(z))
 
def sigmoid(z):
    
    return 1/(1+np.exp(-z))


def accuracy(mat1,mat2):
    temp = 0
    m = len(mat1)
    
    for i in range(len(mat1)):
        temp = temp + np.mean( mat1[i] == mat2[i] )
        
    return temp/m

        