#-*-coding: utf-8-*-
'''
Created on 7 sept. 2012

@author: pierre
'''
import random
import numpy as np

def cross_validation_indices(nbTrialsPerClass,
                             nbTrialsUsed=None,
                             nbValidationTrialsPerClass=1):
    '''Generate the indices that point to the trials in a data test to train and
    test a learning algorithm in a cross validation fashion.
    
    Arguments:
    - nbTrialsPerClass: a mere list of numbers, each number corresponding to the
        number of trials in a given class. So the size of this list gives the
        number of classes.
    - nbTrials: the number of trials of each class used for the cross validation
        It can'fold be higher than the minimum number of trials among all the
        classes (corresponding to the default value if arg set to None).
    - nbValidationTrialsPerClass: the number of trials used for the validation
        for each round.
    
    Example:
    >>> cross_validation_indices({'a':7,'b':9,'c':8},
                                 nbTrialsUsed=6,
                                 nbValidationTrialsPerClass=2)
    [[{'a': [3, 4, 5, 6], 'b': [5, 6, 2, 3], 'c': [4, 6, 2, 1]}, -> Training for the first fold
      {'a': [0, 2], 'b': [0, 7], 'c': [0, 5]}],                  -> Testing for the first fold
      
     [{'a': [5, 6, 0, 2], 'b': [2, 3, 0, 7], 'c': [2, 1, 0, 5]}, -> Training for the second fold
      {'a': [3, 4], 'b': [5, 6], 'c': [4, 6]}],                  -> Testing for the second fold
      
     [{'a': [0, 2, 3, 4], 'b': [0, 7, 5, 6], 'c': [0, 5, 4, 6]}, -> Training for the third fold
      {'a': [5, 6], 'b': [2, 3], 'c': [2, 1]}]]                  -> Testing for the third fold
      
    The same function call will give different outputs, because the trial
        indices are shuffled.
    '''    
    nbClasses = len(nbTrialsPerClass)
    
    minNbTrials = 9999999999
    for class_ in nbTrialsPerClass:
        if nbTrialsPerClass[class_] < minNbTrials:
            minNbTrials = nbTrialsPerClass[class_]
    
    if nbTrialsUsed is None:
        # If nbTrials is not specified, default option is to take the minimum number
        # of trials among all the classes
        nbTrialsUsed = minNbTrials
    elif nbTrialsUsed > minNbTrials:
        msg = "nbTrials is higher than the minimum number of trials per phoneme"
        raise ValueError, msg
    
    trialIndices = {}
    for class_ in nbTrialsPerClass:
        trialIndices[class_] = range(nbTrialsPerClass[class_])
        random.shuffle(trialIndices[class_])
        while len(trialIndices[class_]) > nbTrialsUsed:
            trialIndices[class_].pop()
    
    nbFold = int(nbTrialsUsed / nbValidationTrialsPerClass)
    if nbTrialsUsed % nbValidationTrialsPerClass != 0:
        nbFold += 1
    
    indices = []
    for fold in range(nbFold):
        indices.append([]) # Append a fold
        indices[fold].append({}) # Append training dictionary
        indices[fold].append({}) # Append testing dictionary
        for class_ in nbTrialsPerClass:
            # Testing
            indices[fold][1][class_] = []
            for trial in range(nbValidationTrialsPerClass):
                indices[fold][1][class_].append(trialIndices[class_][trial])
                
            # Training
            indices[fold][0][class_] = []
            for trial in range(nbValidationTrialsPerClass, nbTrialsUsed):
                indices[fold][0][class_].append(trialIndices[class_][trial])
        
        for class_ in nbTrialsPerClass:
            for trial in range(nbValidationTrialsPerClass):
                tmp = trialIndices[class_].pop(0)
                trialIndices[class_].append(tmp)
    
    return indices

################################################################################
########################### PERFORMANCE EVALUATION #############################

def average_in_list(readoutActivity, testingBounds=None):
    '''Return the maximum activities for each trial and for each readout neuron
    (phoneme)
    
    Arguments:
    - readoutActivity: the readout activity of a trial
    - testingBounds: The bounds within which the maximum activity will be
        computed
    '''
    
    if testingBounds == None:
        meanActivities = np.mean(readoutActivity, 0)
    else:
        meanActivities = np.mean(readoutActivity[testingBounds[0]:testingBounds[1], : ], 0)
                
    return meanActivities

def max_activity(readoutActivity, testingBounds=None):
    '''Return the maximum activities for each trial and for each readout neuron
    (phoneme)
    
    Arguments:
    - readoutActivity: the readout activity of a trial
    - testingBounds: The bounds within which the maximum activity will be
        computed
    '''
    
    if testingBounds == None:
        maxActivities = np.max(readoutActivity, 0)
    else:
        maxActivities = np.max(readoutActivity[testingBounds[0]:testingBounds[1], : ], 0)
                
    return maxActivities

def is_correct(activities, correctOutput):
    '''Determine if the current choice based on activities is correct or not
    
    Arguments:
    - activities: a list whose each element represents the activity of a readout
        neuron. These can be obtained with max activity and mean activity...
    - correctOutput: the number of the neuron that should have the highest acti-
        vity
    
    Returns:
    - correct: 1 if correct choice, 0 if not
    - choices: choices of the readout
    '''
    choice = activities.argmax()
    if choice == correctOutput:
        correct = 1
    else:
        correct = 0
    
    return correct, choice

def get_perfs_class(readoutActivity,
                    desOutClass,
                    choiceMethod,
                    testingLimits=None):
    '''Return the performance of the reservoir for a given list of readout
    activity
    
    Arguments:
    - readoutActivity: the activity of the readout (in matrices) in a list
    - desOutClass: the class of the desired output in the same order as given
        by readoutActivity
    - choiceMethod: string - one of ['max', 'mean'] -> the method used to get
        the choice of the readout like average_in_list, max_activity...
    - testingLimits: the limits within which the activity will used to get the
        performances
    
    Return:
    - a list of int for each trial, with 1 for a correct answer and 0 otherwise
    - a list with element being the choice of the readout
    '''
    methods = {'max': max_activity,
               'mean': average_in_list}
    
    method = methods[choiceMethod]
    activities = []
    for trialAct in readoutActivity:
        activities.append(method(trialAct[testingLimits[0]: testingLimits[1]]))
    
    correct = []
    choice = []
    for trialAct, correctClass in zip(activities, desOutClass):
        correctTmp, choiceTmp = is_correct(trialAct, correctClass)
        correct.append(correctTmp)
        choice.append(choiceTmp)
    
    return correct, choice

#def cross_validation(self,
#                     trialsPerClass,
#                     desiredOutputPerClass,
#                     performanceFunc,
#                     nbOut=2,
#                     trainParams={'regression':'ridge',
#                                  'ridgeParam':0.0001,
#                                  'bias':True}):
#    '''Runs a cross validation
#    
#    Arguments
#    - trialsPerClass: a dictionary whose keys are the name of the class and
#        values are the list of data trials
#    - desiredOutputPerClass: a dictionary with the name of each class as key
#        and the desired output used to train and test as value
#    - nbOut: the number of trials out in each class used for validation
#    '''
#    nbTrialsPerClass = {}
#    for class_ in trialsPerClass:
#        try:
#            assert len(trialsPerClass[class_]) == len(desiredOutputPerClass[class_])
#        except KeyError:
#            msg = 'trialsPerClass and desiredOutputPerClass must have the same keys (same classes)'
#            raise Exception, msg
#        except AssertionError:
#            msg = 'The number of trials in trialsPerClass and desiredOutputPerClass must be the same for each class'
#            raise Exception, msg
#        else:
#            nbTrialsPerClass[class_] = len(trialsPerClass[class_])
#    
#    crossvalIndices = cross_validation_indices(nbTrialsPerClass,
#                                               nbValidationTrialsPerClass=nbOut)
#    
#    perfsAllFold = []
#    
#    for fold in crossvalIndices:
#        pass
