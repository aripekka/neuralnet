from __future__ import division
from numpy import exp, isnan, zeros, array, reshape,concatenate,dot
from numpy.random import normal


def sigmoid(x):
    x = 1/(1+exp(-x))
    x[isnan(x)] = 0
    return x

class Layer:
    def __init__(self,n_input):

        self.n_input = n_input
        self.weight_matrix = zeros((0,n_input))
        self.bias_vector = zeros((0,1))
        
    def add_neuron(self,weights=None,bias=None):
        
        if not weights == None:
            #reshape to 1d array
            weights = array(weights)
            weights = reshape(weights,(1,-1))
        else:
            weights = normal(size = (1,self.n_input))

        if not bias == None:            
            bias = array(bias)
            bias = reshape(bias,(1,1))
        else:
            bias = normal(size = (1, 1))
        
        #add neuron (i.e. its weights and bias to the appropriate matrices)
        self.weight_matrix = concatenate((self.weight_matrix, weights))
        self.bias_vector = concatenate((self.bias_vector, bias))
      
    def n_neurons(self):
        return self.weight_matrix.shape[0]
        
    def compute_output(self,input_vector):      
        return sigmoid(dot(self.weight_matrix, input_vector) + self.bias_vector)

    def set_neuron(self,n,weights,bias):
        self.weight_matrix[n,:] = weights
        self.bias_vector[n] = bias

    def get_neuron(self,n):
        return self.weight_matrix[n,:], self.bias_vector[n]
       
    def __repr__(self):
        return self.weight_matrix.__repr__() + '\n' + self.bias_vector.__repr__()
