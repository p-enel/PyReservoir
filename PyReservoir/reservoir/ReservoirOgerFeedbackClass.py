#-*-coding: utf-8-*-
'''
Created on 26 juil. 2013

@author: pierre
'''
import pylab as pl

from ReservoirOgerNodes import ReservoirOger

class ReservoirOgerFeedback(ReservoirOger):
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
        self.feedbackWeights = feedbackWeights
        self.inputOnlyWeights = inputWeights
        newInputWeights = pl.concatenate((inputWeights, feedbackWeights), axis=1)
        
        super(ReservoirOgerFeedback, self).__init__(newInputWeights,
                                                    internalWeights,
                                                    readoutSize,
                                                    *args,
                                                    **kwargs)
    
    def concatenate_inputweights(self):
        '''Re-concatenate input neurons weights and feedback weights'''
        self.inputWeights = pl.concatenate((self.inputOnlyWeights,
                                            self.feedbackWeights), axis=1)
    
    def train(self,
              inputList,
              desiredOutputList,
              fakeFeedback=None,
              **kwargs):
        '''
        '''
        if fakeFeedback is not None:
#             print "len(fakeFeedback)", len(fakeFeedback)
#             print "len(inputList)", len(inputList)
            assert len(fakeFeedback) == len(inputList)
            assert fakeFeedback[0].shape == desiredOutputList[0].shape
            feedbackInput = fakeFeedback
        else:
            feedbackInput = desiredOutputList
        
        newInputList = [pl.concatenate((input_, feedback), axis=1) for input_, feedback in zip(inputList, feedbackInput)]
        
        super(ReservoirOgerFeedback, self).train(newInputList, desiredOutputList, **kwargs)
    
    def train_with_error(self,
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
        
        super(ReservoirOgerFeedback, self).train_with_error(newInputList, desiredOutputList, **kwargs)
    
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
            
            return super(ReservoirOgerFeedback, self).execute(newInput, **kwargs)
        
        else:
            nbTimeSteps = input_.shape[0]
            readout = pl.zeros((nbTimeSteps, self.readoutSize))
            resActivity = pl.zeros((nbTimeSteps, self.nbNeurons))
            self.computeReadout()
            for inputOneTSid, inputOneTS in enumerate(input_):
                newInput = pl.concatenate((inputOneTS, self.readout))[None,:]
                readoutTmp, resActivityTmp = super(ReservoirOgerFeedback, self).execute(newInput, **kwargs)
                readout[inputOneTSid, :] = readoutTmp
                resActivity[inputOneTSid, :] = resActivityTmp
            
            return readout, resActivity
    
    
                