# -*- coding: utf-8 -*-
'''
Created on 2 sept. 2012

@author: pier
'''
# Std imports
import numpy as np
import pylab as pl
import pstats, cProfile, os

from PyReservoir import Reservoir, generateRandomWeights
# Custom imports
#from reservoirClass import Reservoir
#from weightGeneration import generateRandomWeights


if __name__ == '__main__':
    
    #===========================================================================
    # Stims and outputs
    #===========================================================================

    os.putenv('OMP_NUM_THREADS', '1')

    # Time constants
    nbStepsStim = 100
    tau = nbStepsStim / 2
    
    # Input parameters
    input_sparseness = 0.01
    inputSeed = 5
    inputScaling = 1
    
    # Reservoir parameters
    nbNeurons = 500
    reservoir_sparseness = 0.1
    spectral_radius = 1
    reservoirSeed = 6
    outputFunction = np.tanh
#    feedback = False
    weightsVariance = 1
    
    # Readout
    readoutType = 'ridge' # only ridge so far
    ridgeParam = 0.0001
    washout = 0
    
    # Save parameters
    savePath = '/home/pier'
    fileName = 'res.pk'
    
    # Stims and outputs
    
    stim1 = np.concatenate((np.ones((nbStepsStim, 1)), np.zeros((nbStepsStim, 3))), axis=1)
    
    stim2 = np.concatenate((np.zeros((nbStepsStim,1)), np.ones((nbStepsStim, 1)), np.zeros((nbStepsStim, 2))),axis=1)
    stim3 = np.concatenate((np.zeros((nbStepsStim, 2)), np.ones((nbStepsStim, 1)), np.zeros((nbStepsStim,1))),axis=1)
    stim4 = np.concatenate((np.zeros((nbStepsStim, 3)), np.ones((nbStepsStim, 1))),axis=1)
    interStim = np.zeros((nbStepsStim,4))
    
    sequence1 = np.concatenate((stim1, interStim, stim2, interStim, stim3), axis=0)
    desOutSeq1 = np.zeros((sequence1.shape[0], 2))
    desOutSeq1[-nbStepsStim/2:, 0] = 1
    
    sequence2 = np.concatenate((stim4, interStim, stim2, interStim, stim3), axis=0)
    desOutSeq2 = np.zeros((sequence1.shape[0], 2))
    desOutSeq2[-nbStepsStim/2:, 1] = 1
    
    interSequence = np.zeros((nbStepsStim * 2, 4))
    interSequenceOut = np.zeros((nbStepsStim * 2, 2))
    
    # Single stim
    #inputStim = np.concatenate((interSequence, stim1, interSequence, interSequence, interSequence, interSequence, interSequence), axis=0)
    #desOutput = np.zeros((inputStim.shape[0], 1))
    
    # Two stims
    #inputStim = np.concatenate((interSequence, stim1, interStim, stim2, interSequence, interSequence, interSequence, interSequence), axis=0)
    #desOutput = np.zeros((inputStim.shape[0], 1))
    
    # Three stims
#    inputStim = np.concatenate((interSequence, stim1, interStim, stim2, interStim, stim3, interSequence, interSequence, interSequence), axis=0)
#    desOutput = np.zeros((inputStim.shape[0], 1))
    
    # One sequence
    inputStim = np.concatenate((interSequence, sequence1, interSequence, interSequence), axis=0)
    desOutput = np.concatenate((interSequenceOut, desOutSeq1, interSequenceOut, interSequenceOut), axis=0)
    
    # Two sequences
    #inputStimList = []
    #inputStimList.append(np.concatenate((interSequence, sequence1, interSequence), axis=0))
    #inputStimList.append(np.concatenate((interSequence, sequence2, interSequence), axis=0))
    ##inputStim = np.concatenate((interSequence, sequence1, interSequence, sequence2, interSequence), axis=0)
    #desOutputList = []
    #desOutputList.append(np.concatenate((interSequenceOut, desOutSeq1, interSequenceOut), axis=0))
    #desOutputList.append(np.concatenate((interSequenceOut, desOutSeq2, interSequenceOut), axis=0))
    
    inputStimList = [inputStim]
    desOutputList = [desOutput]
    
    # Weight matrices
    
    inputDim = inputStimList[0].shape[1]
    np.random.seed(inputSeed)
    wIn = np.random.randint(0, 2, (nbNeurons, inputStimList[0].shape[1])) * 2 - 1
    
    w = generateRandomWeights(nbUnitsIN=nbNeurons,
                              sparseness=reservoir_sparseness,
                              distribution='normal',
                              spectralRadius=spectral_radius,
                              seed=reservoirSeed)
    
    # Reservoir itself
    
    reservoir = Reservoir(wIn,
                          w,
                          readoutSize=3,
#                          feedbackWeights=None,
                          outputFunction=outputFunction,
                          tau=tau,
                          readoutWeights=None,
                          scipySparse=True)
    
    reservoir.train(inputStimList,
                    desOutputList,
                    regression=readoutType,
                    washout=washout,
                    subsetNodes=range(50),
                    ridgeParam=ridgeParam)
    
    testOut, states = reservoir.execute(inputStimList[0])
    
    if True:
        pl.figure()
        pl.suptitle('One stimulus with spectral radius 1.5')
        pl.subplot(311)
        #pl.plot(np.concatenate((inputStimList[0], inputStimList[1]), axis=0))
        pl.plot(inputStimList[0])
        v = pl.axis()
        x_start = v[0]
        x_end = v[1]
        pl.axis((x_start,x_end,0,1.1))
        pl.subplot(312)
        pl.plot(states)
        #pl.subplot(313)
        #pl.plot(np.concatenate((desOutputList[0], desOutputList[1]), axis=0))
        #v = pl.axis()
        #x_start = v[0]
        #x_end = v[1]
        #pl.axis((x_start,x_end,0,1.1))
        pl.subplot(313)
        pl.plot(testOut)
        pl.show()
    