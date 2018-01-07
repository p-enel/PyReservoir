#-*-coding: utf-8-*-
'''
Created on 6 sept. 2012

@author: pierre
'''
# Standard imports
from __future__ import division
import multiprocessing, sys, traceback, os
from IPython.parallel import Client, require

import pickle as pk
from organize_files import organizeFiles, set_optimization_dir
from parameter_sets import generateSets

def functionJob(arguments):
        
    try:
        parameterSet = arguments[0]
        resFuncArgs = arguments[1]
        
        fullPath = organizeFiles(parameterSet, functionJob.dataDir)
        
        fileName = fullPath.split('/')[-1]
        
        if not fileName in os.listdir('.'):
            
            msg = 'testing set: '
            for param in parameterSet.keys():
                if param in ['inst', 'seed']:
                    msg += '%s %d, '%(param, parameterSet[param])
                else:
                    paramValue, paramFormat = parameterSet[param][:2]
                    msg += ('%s %s, '%(param, paramFormat))%paramValue
            
            msg = msg[:-2]
        
            print msg
            
            results = functionJob.resFunc(parameterSet, resFuncArgs)
            
            f = open(fullPath, 'w')
            pk.dump(results, f)
            f.close()
        else:
            print 'Parameter set already done'
            
    except Exception:
        try:
            os.system('notify-send "Error in gridsearch %s!!!" -t 0'%functionJob.optimizationName)
            os.system('play -q ~/Musique/sad_tombone.wav')
        except:
            pass
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print '#############################################################################################'
        print 'Error in parameter set:'
        print parameterSet
        print 'Traceback of exception: %s of %s'%(exc_value, str(exc_type))
        traceback.print_tb(exc_traceback, limit=10, file=sys.stdout)
        print '\n'
        raise exc_type, exc_value

def gridsearch(parameters,
               pathToResultsDir,
               optimizationName,
               nbInstances=5,
               nbProcessorUsed=1,
               testingMode=False,
               reservoirFunc=None,
               resFuncArgs=None,
               seed=None,
               notify=False):
    '''User called function to explore parameters of a reservoir
    
    Note that the exploration can be stopped and resumed any time,
    the optimizations done will be saved.
    
    Argument:
    - testingMode: do not use parallel processing to test if the reservoir
        function works well (necessary for debugging!) 
    - seed: to replicate results, set a seed, otherwise let to None
    '''
    
    optimizationRootDirectory = pathToResultsDir + optimizationName
    optimizationDataDirectory = set_optimization_dir(optimizationRootDirectory)
    
    # Generate a list of the parameter sets explored
    paramSets = generateSets(parameters, nbInstances=nbInstances, seed=seed)
    
    nbParamSets = len(paramSets)
    print "Nb of parameters sets: %d"%nbParamSets
    
    # Optimization infos:
    optInfos = {'parameters':       parameters,
                'nbInstances':      nbInstances,
                'optimizationName': optimizationName}
    
    # Saving the parameters variable for future processing
    f = open(optimizationRootDirectory+'/opt_infos.pk','w')
    pk.dump(optInfos, f)
    f.close()
    
    functionJobArguments = [(paramSets[i], resFuncArgs) for i in range(nbParamSets)]
    
    functionJob.resFunc = reservoirFunc
    functionJob.optimizationName = optimizationName
    functionJob.dataDir = optimizationDataDirectory
    
    # Create a pool processes and launch the optimization on these processes
    if testingMode:
        for arguments in functionJobArguments:
            functionJob(arguments)
    else:
        # IPython parallel processing
#         c = Client()
#         lbview = c.load_balanced_view()
#         lbview.map(functionJob, functionJobArguments)
        
        # Multiprocessing package
        pool = multiprocessing.Pool(processes=nbProcessorUsed)
        pool.map(functionJob, functionJobArguments)
    
    if notify:
        try:
            os.system('notify-send "Optimization %s is over!!!" -t 0'%optimizationName)
            os.system('play -q ~/Musique/Ta_Da-SoundBible.com-1884170640.wav')
        except:
            pass
    
