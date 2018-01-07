from __future__ import division
import copy
import numpy as np
from scipy import sparse

class Reservoir(object):
    '''A reservoir like neural network
    
    Allows you to train and execute with trained weights a reservoir
    
    Arguments:
    - inputWeights: the matrix of input connections, must be consistent with the
        dimension of the input fed to the network.
    - internalWeights: the matrix of internal connections. For useful activity
        the this matrix chould be sparse and have a spectral radius close to 1.
    - readoutSize: number of neurons in the readout layer. Must be consistent
        with the output given for training.
    - outputFunction: the transfer function of the neurons in the network.
        Default value np.tanh works well with reservoirs.
    - tau: the time constant of the neurons. The neurons are leaky integrators,
        but the default value 1 set the neurons to a non-leaky regime.
    - readoutWeights: optional. If the network has already been trained or to
        test the network with specific readout weights, you can specify it here.
    - scipySparse: especially for reservoirs with a big number of neurons
        (~ 1500, to be confirmed...) the use of scipy.sparse module accelerate
        the processing, this argument should then be set to True.
    '''
    def __init__(self,
                 inputWeights,
                 internalWeights,
                 readoutSize,
#                 feedbackWeights=None,
                 outputFunction=np.tanh,
                 tau=1,
                 readoutWeights=None,
                 scipySparse=False):
        
        self.scipySparse = scipySparse
        
        if scipySparse:
            self.inputWeights = sparse.csr.csr_matrix(inputWeights)
            self.internalWeights = sparse.csr.csr_matrix(internalWeights)
            self.weight_inputs = self._weight_inputs_sparse
        else:
            self.inputWeights = inputWeights
            self.internalWeights = internalWeights
            self.weight_inputs = self._weight_inputs
        
        self.nbNeurons = internalWeights.shape[0]
        
        self.states = np.zeros(self.nbNeurons)
        self.output = np.zeros(self.nbNeurons)
        
        self.readoutSize = readoutSize
        self.outputFunction = outputFunction
#        self.inAndFbWeights = np.concatenate((self.inputWeights, self.feedbackWeights), axis=1)
        self.tau = tau
        self.tau_inv = 1. / tau
        
        if readoutWeights is not None:
            print 'readoutSize : %d'%readoutSize
            print 'readoutWeights.shape[0] : %d'%readoutWeights.shape[0]
            if readoutWeights.shape[1] != readoutSize:
                raise ValueError, 'The readout weights must have a number of columns equal to readoutSize'
            else:
                self.readoutWeights = readoutWeights
        
#        self.feedbackWeights = feedbackWeights
#        if feedbackWeights is None:
#            self.feedbackWeights = np.zeros((self.nbNeurons, self.readoutSize))
        
        self.readout = None
        self.readoutWeights = readoutWeights
    
    def _weight_inputs(self, *args, **kwargs):
        raise NotImplementedError
    
    def _weight_inputs_sparse(self, *args, **kwargs):
        raise NotImplementedError
    
    def train(self,*args, **kwargs):
        raise NotImplementedError
            
    def reset(self):
        '''Reset the values of states and output to zero'''
        self.set_output(0)
        self.set_readout(0)
    
    def computeOutput(self):
        '''Compute output from state with transfer function'''
        self.output = self.outputFunction(self.states)
    
    def computeReadout(self):
        '''Compute readout from output with readout weights'''
        outputWithBias = np.concatenate((self.output,[1]))
        self.readout = np.dot(outputWithBias, self.readoutWeights)
    
    def set_output(self, outputValue):
        '''Set the output of the reservoir neurons
        
        Arguments:
        - it can either be a single real value or a vector the size of the
            reservoir
        '''
        if type(outputValue) not in [float, int]:
            if type(outputValue) == np.ndarray and outputValue.shape == (self.nbNeurons,):
                self.output = outputValue
            else:
                raise ValueError, 'The argument for set_output method is not valid'
        else:
            self.output[:] = outputValue
            
    def set_readout(self, readoutValue):
        '''Set the readout of the reservoir
        
        Arguments:
        - it can either be a single real value or a vector the size of the
            reservoir'''
        if type(readoutValue) not in [float, int]:
            if type(readoutValue) == np.ndarray and readoutValue.shape == (self.readoutSize,):
                self.readout = readoutValue
            else:
                raise ValueError, 'The argument for set_output method is not valid'
        else:
            self.readout = np.ones(self.readoutSize) * readoutValue
    
    def execute(self, *args, **kwargs):
        raise NotImplementedError
    
    def execute_list(self, *args, **kwargs):
        raise NotImplementedError
