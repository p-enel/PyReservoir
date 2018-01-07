#-*-coding: utf-8-*-
'''
Created on 10 déc. 2012

@author: pierre
'''

import matplotlib.pyplot as pl
import numpy as np

from plotting import add_vertical_lines, add_vertical_shaded_areas

def vis_test_class(readoutActivity,
                   desoutClass,
                   choice,
                   learningLimits,
                   desiredOutput=None,
                   eventTimeId=None,
                   testList=None,
                   reservoirActivity=None,
                   resNeuronToShow=None):
    '''Visualize the readout activity (and others..) with colored background for
    the learning window
    
    Arguments:
    - readoutActivity: the activity of the readout in a list
    - desoutClass: the desired output class for each input
    - choice: a list of int corresponding to the choice of the readout
    - eventTimeId: an int - the index of the event (for decoding)
    - testList: the input list for testing, if left to None, the inputs are not
        shown
    - reservoirActivity: the activity of the reservoir in a list if it must be
        shown otherwise it must left to None
    - resNeuronToShow: list of the indices of the reservoir neurons that must be
        plotted!
        
    Returns nothing but a beautiful figure!'''
    
    classId = {}
    for class_ in desoutClass:
        if class_ not in classId:
            classId[class_] = None
    for index, class_ in enumerate(classId):
        classId[class_] = index
            
    colorDic = {0:'b', 1:'g', 2:'r', 3:'c', 4:'m', 5:'y'}
    
    # Get the number of classes:    
    classes = []
    for desOut in desoutClass:
        if desOut not in classes:
            classes.append(desOut)
    
    nbClasses = len(classes)
    
    if nbClasses > 6:
        msg = 'There are only 6 different colors for the classes, others and '
        msg += 'there are %d classes. Other colors must be defined!'%nbClasses
        raise NotImplementedError, msg
    
    startLim, endLim = learningLimits
    
    nbTrials = len(readoutActivity)
    nbTimeSteps, nbReadout = readoutActivity[0].shape
    
    readoutForPlot = np.concatenate(readoutActivity, axis=0)
    
    nbAxes = 1
    if testList is not None:
        inputForPlot = np.concatenate(testList, axis=0)
        nbAxes += 1
    if reservoirActivity is not None:
        if resNeuronToShow is None:
            resForPlot = np.concatenate(reservoirActivity, axis=0)
        else:
            resActivity = [reservoirActivity[trialId][:,resNeuronToShow] for trialId in range(nbTrials)]
            resForPlot = np.concatenate(resActivity, axis=0)
        nbAxes += 1
    if desiredOutput is not None:
        desiredOutToPlot = np.concatenate(desiredOutput, axis=0)
        nbAxes += 1
    
    # Create figure and subplots according to the arguments
    fig = pl.figure()
    
    axes = []
    currentSubplot = 1
    if testList is not None:
        axes.append(pl.subplot(nbAxes, 1, 1))
        pl.plot(inputForPlot)
        currentSubplot += 1
    
    if reservoirActivity is not None:
        if testList is not None:
            sharex=axes[0]
        else:
            sharex=None
        axes.append(pl.subplot(nbAxes, 1, currentSubplot, sharex=sharex))
        pl.plot(resForPlot)
        currentSubplot += 1
    
    if desiredOutput is not None:
        if currentSubplot != 1:
            sharex=axes[0]
        else:
            sharex=None
        axes.append(pl.subplot(nbAxes, 1, currentSubplot, sharex=sharex))
        pl.plot(desiredOutToPlot)
        currentSubplot += 1
    
    if reservoirActivity is not None or testList is not None:
        sharex=axes[0]
    else:
        sharex=None
    
    axes.append(pl.subplot(nbAxes, 1, currentSubplot, sharex=axes[0]))
        
    pl.plot(readoutForPlot)
    
    for trialId in range(nbTrials):
        trainingLimsTmp = [startLim + (trialId * nbTimeSteps), endLim + (trialId * nbTimeSteps)]
        endOfTrial = (trialId + 1) * nbTimeSteps
        
        for ax in axes:
            # Thick line for the end of the trial
            add_vertical_lines([endOfTrial], ax=ax, color='k', linewidth=2)
            # Background shaded area to show the class of the input and the
            # training limits
            add_vertical_shaded_areas([trainingLimsTmp],
                                      ymin=0,
                                      ymax=0.95,
                                      facecolor=colorDic[classId[desoutClass[trialId]]],
                                      alpha=0.15)
            # Upper band on the same position that the above shaded area to
            # notify the choice of the readout
            add_vertical_shaded_areas([trainingLimsTmp],
                                      ymin=0.95,
                                      ymax=1,
                                      facecolor=colorDic[classId[choice[trialId]]],
                                      alpha=0.15)
            
        
    if eventTimeId is not None:
        eventPositions = [eventTimeId + (trialId * nbTimeSteps) for trialId in range(nbTrials)]
        for ax in axes:
            # Plot event line
            add_vertical_lines(eventPositions, ax=ax, color='k', linestyle='--')
    
    if axes[0].get_xlim()[1] > 1000:
        axes[0].set_xlim([0, 1000])
    
    fig.canvas.manager.toolbar.pan()
    
    return fig