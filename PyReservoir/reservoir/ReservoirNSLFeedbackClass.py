#-*-coding: utf-8-*-
'''
Created on 26 juil. 2013

@author: pierre
'''
import pylab as pl

from ReservoirNSLNeurons import ReservoirNSL

class ReservoirNSLFeedback(ReservoirNSL):
    '''
    '''
    
    def __init__(self,
                 inputWeights,
                 internalWeights,
                 readoutSize,
                 feedbackWeights=None,
                 *args, **kwargs):
        '''
        '''
        if feedbackWeights is None:
            raise ValueError, "No feedback weights was passed as argument!..."
        newInputWeights = pl.concatenate((inputWeights, feedbackWeights), axis=1)
        super(ReservoirNSLFeedback, self).__init__(newInputWeights,
                                                internalWeights,
                                                readoutSize,
                                                *args,
                                                **kwargs)
    
    def train(self,
              inputList,
              desiredOutputList,
              fakeFeedback=None,
              **kwargs):
        '''
        '''
        if fakeFeedback is not None:
            assert len(fakeFeedback) == len(inputList)
            assert fakeFeedback[0].shape == desiredOutputList[0].shape
            feedbackInput = fakeFeedback
        else:
            feedbackInput = desiredOutputList
        
        newInputList = [pl.concatenate((input_, feedback), axis=1) for input_, feedback in zip(inputList, feedbackInput)]
        
        super(ReservoirNSLFeedback, self).train(newInputList, desiredOutputList)
    
    def execute(self,
                input_,
                fakeFeedback=None,
                **kwargs):
        '''
        '''
        if fakeFeedback is not None:
            assert fakeFeedback.shape[0] == input_.shape[0]
            assert fakeFeedback.shape[1] == self.readoutSize
            newInput = pl.concatenate((input_, fakeFeedback), axis=1)
            
            return super(ReservoirNSLFeedback, self).execute(newInput, **kwargs)
        
        else:
            nbTimeSteps = input_.shape[0]
            readout = pl.zeros((nbTimeSteps, self.readoutSize))
            resActivity = pl.zeros((nbTimeSteps, self.nbNeurons))
            self.computeReadout()
            for inputOneTSid, inputOneTS in enumerate(input_):
                newInput = pl.concatenate((inputOneTS, self.readout))[None,:]
                readoutTmp, resActivityTmp = super(ReservoirNSLFeedback, self).execute(newInput, **kwargs)
                readout[inputOneTSid, :] = readoutTmp
                resActivity[inputOneTSid, :] = resActivityTmp
            
            return readout, resActivity
    
    
                