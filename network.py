from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from layer import Layer
import random

from time import clock

random.seed()

class Network:
    def __init__(self,neuron_count_list = [1,1,1]):

        self.layer_list = []
       
        for i in xrange(1,len(neuron_count_list)):
            #new layer with input count determined by the prev layer neuron count
            new_layer = Layer(neuron_count_list[i-1])
            
            for j in xrange(neuron_count_list[i]):
                new_layer.add_neuron(weights=None,bias=None)
            
            self.layer_list.append(new_layer)
            
    def compute_outputs(self, input_vector):
        
        input_vector = np.array(input_vector).reshape((-1,1))
        if not input_vector.size == self.layer_list[0].n_input:
            raise ValueError('Input vector element count does not match the number of input neurons!')

        outputs = [input_vector]   
        for layer in self.layer_list:
            output_vector = layer.compute_output(input_vector)
            outputs.append(output_vector)
            input_vector = output_vector
        
        return outputs

    def compute_outputs_nocheck(self, input_vector):
        n_layers = len(self.layer_list)
        layers = self.layer_list

        outputs = [0]*(n_layers+1)
        outputs[0] = input_vector
        for i in xrange(n_layers):
            input_vector = layers[i].compute_output(input_vector)
            outputs[i+1] = input_vector
        return outputs

    def train_slow(self,training_set,cycles,learning_rate = 1,bias_rate=0.5):
        '''
        Trains the network using backpropagation. 
        OLD VERSION: ~20% SLOWER, ONLY FOR ILLUSTRATIVE PURPOSES!
        '''

        #init delta, deltaw and deltab lists
        delta_list = []
        dw_list = []
        db_list = []

        for layer in self.layer_list:
            delta_list.append(np.zeros(layer.n_neurons()))
            dw_list.append(np.zeros(layer.weight_matrix.shape))                
            db_list.append(np.zeros(layer.bias_vector.shape))
        
        for i in xrange(cycles):
            for datapoint in training_set:
                                      
                input_vector = np.array(datapoint[0]).reshape((-1,1))
                ideal_output = np.array(datapoint[1]).reshape((-1,1))
               
                init_output = self.compute_outputs(input_vector)                
                
                error = init_output[-1] - ideal_output


                #compute the output layer deltas and deltaws
                delta_list[-1] = error*init_output[-1]*(1-init_output[-1])
              
                dw_list[-1] = -learning_rate * np.dot(init_output[-2], delta_list[-1].T)
                dw_list[-1] = dw_list[-1].T                                               

                #compute the output layer deltab
                db_list[-1] = -bias_rate * delta_list[-1]                                          


                #compute the inner layer deltas and deltaws
                for l_ind in xrange(len(self.layer_list)-2,-1,-1):
                    w = self.layer_list[l_ind+1].weight_matrix
                                   
                    sums = np.dot(delta_list[l_ind+1].T, w)
                                       
                    delta_list[l_ind] = sums.T*init_output[l_ind+1]*(1-init_output[l_ind+1])
                
                    dw_list[l_ind] = -learning_rate * np.dot(init_output[l_ind], delta_list[l_ind].T)                                                
                    dw_list[l_ind] = dw_list[l_ind].T

                    #deltab                   
                    db_list[l_ind] = -bias_rate * delta_list[l_ind]      

                #adjust neurons
                for l_ind in xrange(len(self.layer_list)):
                    layer = self.layer_list[l_ind]
                    layer.weight_matrix = layer.weight_matrix + dw_list[l_ind] 
                    layer.bias_vector = layer.bias_vector + db_list[l_ind] 
       

    def train(self,training_set,cycles,learning_rate = 1,bias_rate=0.5):
        '''
        Trains the network using backpropagation
        '''

        #local references for speed up
        network_outputs = self.compute_outputs_nocheck
        layers = self.layer_list   
        n_layers = len(self.layer_list)
        dot=np.dot

        #Dataformat check and conversion
        checked_data = [0]*len(training_set)

        for i in xrange(len(training_set)):
            checked_data[i] = (np.array(training_set[i][0]).reshape((-1,1)), 
                               np.array(training_set[i][1]).reshape((-1,1)))
   
        for i in xrange(cycles):
            for datapoint in checked_data:

                init_output = network_outputs(datapoint[0])                
                
                #compute the output layer deltas and deltaws and make adjustments
                delta = (init_output[-1] - datapoint[1])*init_output[-1]*(1-init_output[-1])
              
                layers[-1].weight_matrix -= learning_rate * dot(delta, init_output[-2].T)
                layers[-1].bias_vector -= -bias_rate * delta 

                #compute the inner layer deltas and deltaws and make adjustments
                for l_ind in xrange(n_layers-2,-1,-1):
 
                    delta = dot(layers[l_ind+1].weight_matrix.T, delta)*init_output[l_ind+1]*(1-init_output[l_ind+1])

                    layers[l_ind].weight_matrix -= learning_rate * dot(delta, init_output[l_ind].T)
                    layers[l_ind].bias_vector -= bias_rate * delta
          


    def __repr__(self):
        string = ''
        for i in self.layer_list:
            string = string + i.__repr__() + '\n'

        return string

