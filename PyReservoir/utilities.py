#-*-coding: utf-8-*-
'''
Created on 6 sept. 2012

@author: pierre
'''
#import pickle as pk

def parse_readout_activity(readoutActivity,
                           nbTSperTrial=None,
                           nbTrials=None):
    '''Parse the readout activity into the different trials
    
    Arguments:
    One of nbTSperTrial or nbTrials must be defined.
    '''
    if nbTSperTrial == None and nbTrials == None:
        raise ValueError, 'You must either specify nbTSperTrial OR nbTrials'
    
    parsedActivity = []
    if nbTSperTrial is not None:
        nbTrials = readoutActivity.shape[0] / nbTSperTrial
    elif nbTrials is not None:
        nbTSperTrial = readoutActivity.shape[0] / nbTrials
    else:
        raise Exception, 'Argument missing here!'
    
    for i in range(nbTrials):
        parsedActivity.append(readoutActivity[i*nbTSperTrial : (i+1) * nbTSperTrial])
    return parsedActivity

def get_training_bounds(signalAlignedOn, nbTSperSec, windowCenter, windowSize):
    '''Convert the windowCenter and windowSize variables expressed in ms in
    training bounds corresponding to time steps in the input data.
    
    Arguments:
    - signaledAlignedOn: if the signal is aligned on en event, the variables
        windowCenter and windowSize are expressed relative to this reference
    - nbTSperSec: number of time steps per second to make the conversion
    - windowCenter and windowSize: the center and half the size of the training
        window
    '''
    wCenterInTS = windowCenter / 1000 * nbTSperSec
    wSizeInTS = windowSize / 1000 * nbTSperSec
    alignmentInTS = signalAlignedOn / 1000 * nbTSperSec
    
    infBoundTraining = alignmentInTS + wCenterInTS - wSizeInTS
    supBoundTraining = alignmentInTS + wCenterInTS + wSizeInTS
    trainingBounds = [infBoundTraining, supBoundTraining]
    return trainingBounds

#def set_results_directory():
#    ok = False
#    while not ok:
#        resDir = raw_input('Type the full path of the results directory and press enter:\n')
#        print 'The results of grid search explorations will be in'
#        print '%s/optimizationName'%resDir
#        print 'Make sure there is no space in the path, it may complicate things!'
#        answer = raw_input('Is that ok? [y]/n')
#        if answer != 'n':
#            ok = True
