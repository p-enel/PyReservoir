#-*-coding: utf-8-*-
'''
Created on 16 juin 2012

@author: pierre
'''
# Standard imports
import os
from PyReservoir import Reservoir, gridsearch
'''Here is an example of file to optimize the parameters of a reservoir
'''

os.putenv('OMP_NUM_THREADS', '1')

if __name__ == '__main__':
    
    '''load input data'''
    
    '''############EXAMPLE to be modified############'''
#                  parameter             start     end     step    format      letter
    parameters = {'tau':                 [0,        0,      0.1,    '%.1f',     't'], #log scale
                  'inputSparseness':     [1,        1,      0.1,    '%.1f',     'iSp'],
                  'spectralRadius':      [0,        0,      0.1,    '%.1f',     'rad'], # NOT log scale
                  'weightsVariance':     [0,        0,      0.1,    '%.1f',     'wV'], # log scale
                  'nbNeurons':           [400,      400,    1,      '%d',       'nbN'],
                  'reservoirSparseness': [0,        0,      1,      '%.1f',     'rS'],
                  'inputScaling':        [1.6,      1.6,    0.1,    '%.1f',     'iSc'], # log scale 10^1.6 ~= 40
                  'windowCenter':        [-870,     370,    10,     '%d',       'winC'],
                  'windowSize':          [5,        100,    5,      '%d',       'winS'],
                  'ridgeParam':          [4.5,      4.5,    0.1,    '%.1f',     'rP']} # log scale (-x)
    
    '''############EXAMPLE to be modified############'''
    def reservoirFunc(parameterSet, args):
        'args MUST BE A DICTIONARY WITH THE SAME KEYS AS parameters'
        
        tau = parameterSet['tau'][0]
        inSpar = parameterSet['inputSparseness'][0]
        inScaling = parameterSet['inputScaling'][0]
        resSpar = parameterSet['reservoirSparseness'][0]
        spectralR = parameterSet['spectralRadius'][0]
        nbNeurons = parameterSet['nbNeurons'][0]
        weightsVar = parameterSet['weightsVariance'][0]
        windowCenter = parameterSet['windowCenter'][0]
        windowSize = parameterSet['windowSize'][0]
        ridgeParam = parameterSet['ridgeParam'][0]
        seed = parameterSet['seed']
        
        msg = 'optimization: tau %.1f, inSpar %.1f, inScaling %.1f, '
        msg += 'resSpar %.1f, spectralR %.1f, nbNeurons %d, weightsVar %.1f, ' 
        msg += 'windowCenter %d, windowSize %d, ridgeParam %f, seed %s\n'
        msg = msg%(tau, inSpar, inScaling, resSpar, spectralR, nbNeurons,
                   weightsVar, windowCenter, windowSize, ridgeParam, str(seed))
        
        # Convert the values from the log scale to normal scale
        tau = int(round(10 ** tau))
        inScaling = 10 ** inScaling
        weightsVar = 10 ** weightsVar
        ridgeParam = 10 ** -ridgeParam
        
        reservoir = Reservoir('arguments for this reservoir')
        reservoir.train('Arguments for training')
        reservoirOutput = reservoir.execute('input data for test')
        
        perfs = get_performance(reservoirOutput, 'desiredOutput')
        
        results = {'perfs' : perfs,
                   'otherThingsToSave' : None}
        
        return results
    
    '''############EXAMPLE to be modified############'''
    def get_performance(reservoirOutput, desiredOutput): #arguments can be different!
        '''get the performance of the test'''
        perfs = None # To be Removed
        return perfs
    
    gridsearch(parameters=parameters,
               pathToResultsDir='PATH TO THE OPTIMIZATION FOLDER',
               optimizationName='NAME OF THIS PARTICULAR OPTIMIZATION',
               reservoirFunc=reservoirFunc,
               resFuncArgs='DICTIONARY WITH THE ARGUMENTS FOR THE RESERVOIR FUNCTION',
               nbInstances=5,
               nbProcessorUsed=1,
               testingMode=False,
               seed=None)
