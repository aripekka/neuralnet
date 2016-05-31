from __future__ import division
import numpy as np


def sigmoid(x):
    x = 1/(1+np.exp(-x))
    x[np.isnan(x)] = 0
    return x

class Layer:
    def __init__(self,n_input):

        self.n_input = n_input
        self.weight_matrix = np.zeros((0,n_input))
        self.bias_vector = np.zeros((0,1))
        
    def add_neuron(self,weights=None,bias=None):
        
        if not weights == None:
            #reshape to 1d array
            weights = np.array(weights)
            weights = np.reshape(weights,(1,-1))
        else:
            weights = np.random.normal(size = (1,self.n_input))

        if not bias == None:            
            bias = np.array(bias)
            bias = np.reshape(bias,(1,1))
        else:
            bias = np.random.normal(size = (1, 1))
        
        #add neuron (i.e. its weights and bias to the appropriate matrices)
        self.weight_matrix = np.concatenate((self.weight_matrix, weights))
        self.bias_vector = np.concatenate((self.bias_vector, bias))
      
    def n_neurons(self):
        return self.weight_matrix.shape[0]
        
    def compute_output(self,input_vector):      
        return sigmoid(np.dot(self.weight_matrix, input_vector) + self.bias_vector)

    def set_neuron(self,n,weights,bias):
        self.weight_matrix[n,:] = weights
        self.bias_vector[n] = bias

    def get_neuron(self,n):
        return self.weight_matrix[n,:], self.bias_vector[n]
       
    def __repr__(self):
        return self.weight_matrix.__repr__() + '\n' + self.bias_vector.__repr__()
