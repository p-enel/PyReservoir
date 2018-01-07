#-*-coding: utf-8-*-
'''
Created on 10 déc. 2012

@author: pierre
'''
import numpy as np

def generate_desired_output(desiredOutputNumber,
                            nbOutputNeurons,
                            nbTimeSteps,
                            startTraining):
    '''Generate the desired activity of the readout for one class in the case of
    local representation (one neuron for each class)
    
    Arguments:
    - desiredOutputNumber: the number of the neuron that will be active for this
        class
    - nbOutputNeurons: the total number of readout neurons
    - nbTimeSteps: the total number of time steps from the beginning of the
        input to the end of the training
    - startTraining: the time step when the training should start. The activity
        of the neuron that represent the class will be 0 before and 1 after this
        time step.
    '''
    assert startTraining < nbTimeSteps
    templateOutput = np.zeros((nbTimeSteps, nbOutputNeurons))
    templateOutput[startTraining:, desiredOutputNumber] = 1
    
    return templateOutput
    
def generate_all_desout(nbOutputs,
                        windowLimits):
    '''Generate the desired outputs of each class in the case of a local
    representation (one neuron for each class)
    
    Arguments:
    - nbOutputs: the number of readout neurons, so in other words the number of
        class of in this problem.
    - windowLimits: the lower and upper limits of the training period. The
        readout neurons will be active only during this period.
    
    Returns:
    - desOutputs: a list of the successive desired outputs
    '''
    startTraining, nbTimeSteps = windowLimits
    
    desOutputs = []
    
    for desiredOutputNumber in range(nbOutputs):
        desOutputs.append(generate_desired_output(desiredOutputNumber,
                                                  nbOutputs,
                                                  nbTimeSteps,
                                                  startTraining))
    
    return desOutputs

#def gen_all_desout_by_class(classNames,
#                            windowLimits):
#    '''Generate the desired outputs of each class in the case of a local
#    representation (one neuron for each class)
#    
#    Arguments:
#    - classNames: a list containing the name of each class
#    - windowLimits: the lower and upper limits of the training period. The
#        readout neurons will be active only during this period.
#    
#    Returns:
#    - desOutputs: a dictionary whose keys are the name of each class and the
#        values are desired outputs of each class
#    '''
#    nbClasses = len(classNames)
#    
#    desOutList = generate_all_desout(nbClasses, windowLimits)
#    
#    desOutputs = dict.fromkeys(classNames)
#    
#    for i, class_ in enumerate(classNames):
#        desOutputs[class_] = desOutList[i]
#        
#    return desOutputs

def gen_all_desout_by_class(classNames,
                            windowLimits,
                            dataBefore=None):
    '''Generate the desired outputs of each class in the case of a local
    representation (one neuron for each class)
    
    Arguments:
    - classNames: a list containing the name of each class
    - windowLimits: the lower and upper limits of the training period. The
        readout neurons will be active only during this period.
    - dataBefore: the number of time steps before the training window that will
        be fed to the network. If left to None, all the available data will be
        kept, meaning in this method that all the time steps before the training
        window will have zero has desired output.
    
    Returns:
    - desOutputs: a dictionary whose keys are the name of each class and the
        values are desired outputs of each class
    '''
    classNames.sort()
    nbClasses = len(classNames)
    readoutNeuronsClasses = {}
    
    if dataBefore is not None:
        windowLimits = [dataBefore,
                        windowLimits[1] - windowLimits[0] + dataBefore]
    
    desOutList = generate_all_desout(nbClasses, windowLimits)
    
    desOutputs = dict.fromkeys(classNames)
    
    for i, class_ in enumerate(classNames):
        desOutputs[class_] = desOutList[i]
        readoutNeuronsClasses[i] = class_
        
    return desOutputs, readoutNeuronsClasses

def gen_desout_2classes_one_output(classNames,
                                   windowLimits,
                                   dataBefore=None):
    '''Generate the desired outputs for two classes with one output neuron being
    positive for one class and negative for the other
    
    Arguments:
    - classNames: a list containing the name of each class
    - windowLimits: the lower and upper limits of the training period. The
        readout neurons will be active only during this period.
    - dataBefore: the number of time steps before the training window that will
        be fed to the network. If left to None, all the available data will be
        kept, meaning in this method that all the time steps before the training
        window will have zero has desired output.
    
    Returns:
    - desOutputs: a dictionary whose keys are the name of each class and the
        values are desired outputs of each class
    '''
    classNames.sort()
    assert len(classNames) == 2
    
    if dataBefore is not None:
        windowLimits = [dataBefore,
                        windowLimits[1] - windowLimits[0] + dataBefore]
    
    nbTS = windowLimits[1]
    desOutputs = {}
    desOutputs[classNames[0]] = np.zeros((nbTS,1))
    desOutputs[classNames[0]][windowLimits[0]:windowLimits[1]] = 1
    desOutputs[classNames[1]] = np.zeros((nbTS,1))
    desOutputs[classNames[1]][windowLimits[0]:windowLimits[1]] = -1
    
    return desOutputs
