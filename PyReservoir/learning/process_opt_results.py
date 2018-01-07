# -*- coding: utf-8 -*-
"""
Created on 6 févr. 2012

@author: pier
"""
from __future__ import division
import os, copy, warnings
import numpy as np
import pickle as pk
from parameter_sets import get_val_with_indices, show_parameters

def concatenateResults(optFilesDir, optInfos, funs):
    
    def rec(results, currentSet, funs, parameters, indexRef):
        '''Args:
        - results: a dictionary with empty matrices that will gather the results
        - currentSet: as initial argument, it must be a dictionary with the
            parameters name as keys
        - funs: a list of the function that will retrieve a specific result from
            the results files
        - parameters: the dictionary of the parameters (see optimisation for an
            example
        '''
        
        foldies = os.listdir('.') # foldies for folders or files
        try:
            filePresent = len(foldies[0].split(' ')) == 1
        except:
            print os.getcwd()
            raise Exception, 'Error: there may be optimization files missing, try to rerun the optimization'
            
        if filePresent: #If the first element listed has a name with only one element (it is a result file)
            if not len(foldies) == nbInstances:
                print os.getcwd()
                raise Exception, "The number of files is not corresponding to the number of instances"
            
            indicesList = getIndicesList(currentSet, parameters)
            resList = copy.deepcopy(funs)
            for fun in resList:
                resList[fun] = []
                
            for instFile in foldies:
                
    ############ Change made to adapt to the results of the reservoir decoder#######
                f = open(instFile,'r')
                data = pk.load(f)
                for fun in funs:
                    resList[fun].append(funs[fun](data))
                f.close()
                
#                timeStepForPerfs = 200 + 250
#                
#                file_ = instFile
#                f = open(file_,'r')
#                data = pk.load(f)
#                for fun in funs:
#                    resList[fun] = funs[fun](data)
#                f.close()
                
            for fun in funs:
                try:
                    results[fun]['data'][tuple(indicesList)] = resList[fun]
                    results[fun]['mean'][tuple(indicesList)] = np.mean(resList[fun])
                    results[fun]['std'][tuple(indicesList)] = np.std(resList[fun])
                except IndexError:
                    print 'Error: %s\n%s'%(str(indicesList), str(currentSet))
            
################################################################################
        else: 
            paramName = foldies[0].split(' ')[0]
            for folder in foldies:
                paramValue = folder.split(' ')[1]
                # Test if the parameter value is part of the values defined by parameter
                
#                valueList = [str(value) for value in indexRef[paramName]['val2ind'].keys()]
                valueList = ['%s'%parameters[paramName][3]%value for value in indexRef[paramName]['val2ind'].keys()]
                if paramValue not in valueList:
                    print "Parameter %s with value %s is not part of this optimization"%(str(paramName), str(paramValue))
                else:
                    currentSet[paramName] = float(paramValue)
                    try:
                        os.chdir(folder)
                    except:
                        print 'yo'
                    rec(results, currentSet, funs, parameters, indexRef)
                    os.chdir("..")
    
    def generateIndexRef(parameters):
        paramIndices = {}
        for param in parameters:
            paramIndices[param] = {}
            paramIndices[param]['val2ind'] = {}
            paramIndices[param]['ind2val'] = {}
            paramBounds = parameters[param]
            start = paramBounds[0]
            step = paramBounds[2]
            end = paramBounds[1]
            paramValues = np.arange(start, end + step / 2., step)
            if paramValues[-1] > end:
                paramValues = paramValues[:-1]
            i = 0
            for val in paramValues:
                # This little operation that appears stupid is needed to compare
                # two floats, to understand try 0.3 == 0.2 + 0.1
                precision = 100000000000000
                val = round(val*precision)/precision
                
                paramIndices[param]['val2ind'][val] = i
                paramIndices[param]['ind2val'][i] = val
                i+=1
            
        return paramIndices
    
    parameters = optInfos['parameters']
    nbInstances = optInfos['nbInstances']
    
    # Get the number of dimensions
    dims = []
    for param in parameters:
        start = parameters[param][0]
        end = parameters[param][1]
        step = parameters[param][2]
        paramValues = np.arange(start, end + step / 2., step)
        if paramValues[-1] > end:
            paramValues = paramValues[:-1]
        nbSteps = len(paramValues)
        dims.append(nbSteps)
    
############# Change made to adapt to the results of the reservoir decoder#######
#    dims.append(900) # To save the activity of all the timesteps
################################################################################    
    # Initialize the matrices of results
    results = copy.deepcopy(funs)
    zeroMat = np.zeros(dims)
############ Change made to adapt to the results of the reservoir decoder#######
    for fun in results:
        results[fun] = {'data': np.zeros(dims+[nbInstances]),
                        'mean': zeroMat.copy(),
                        'std': zeroMat.copy()}
        
################################################################################

    # Initialize the temporary variable currentSet to memorize the sets of parameters
    currentSet = copy.deepcopy(parameters)
    for param in currentSet:
        currentSet[param] = None
    
    # Create a dictionary with the indices for correspondence btw params dict
    # and results matrices
    paramsIndex = copy.deepcopy(currentSet)
    paramNames = parameters.keys()
    for i in range(len(paramNames)):
        paramsIndex[paramNames[i]] = i
    
    os.chdir(optFilesDir + "/data")
    indexRef = generateIndexRef(parameters)
    
    rec(results, currentSet, funs, parameters, indexRef)
    
    results['indexRef'] = indexRef
    results['paramsOrder'] = parameters.keys()
    
    return results

def concatenateResults_permutations(optFilesDir, optInfos, funs):
    
    def rec(results, currentSet, funs, parameters, indexRef):
        '''Args:
        - results: a dictionary with empty matrices that will gather the results
        - currentSet: as initial argument, it must be a dictionary with the
            parameters name as keys
        - funs: a list of the function that will retrieve a specific result from
            the results files
        - parameters: the dictionary of the parameters (see optimisation for an
            example)
        '''
        
        foldies = os.listdir('.') # foldies for folders or files
        try:
            filePresent = len(foldies[0].split(' ')) == 1
        except:
            print os.getcwd()
            raise Exception, 'Error: there may be optimization files missing, try to rerun the optimization'
            
        if filePresent: #If the first element listed has a name with only one element (it is a result file)
            if not len(foldies) == nbInstances*nbPermutations:
                print os.getcwd()
#                raise Exception, "The number of files is not corresponding to the number of instances"
            
            indicesList = getIndicesList(currentSet, parameters)
            resList = dict.fromkeys(funs.keys())
            for fun in resList:
                resList[fun] = []
            
            resultsPerm = []
            for perm in range(nbPermutations):
                resultsPerm.append(copy.deepcopy(resList))
            
            for instFile in foldies:
                
                instFileTmp = instFile
                permStr = instFileTmp.split('_')[-2]
                permId = int(permStr.replace('perm',''))
                
                if permId < nbPermutations:
                    f = open(instFile,'r')
                    data = pk.load(f)
                    for fun in funs:
                        try:
                            resultsPerm[permId][fun].append(funs[fun](data))
                        except:
                            print 'yo'
                    f.close()
                
            for fun in funs:
                for permId in range(nbPermutations):
                    try:
                        indicesTmp = tuple(indicesList + [permId])
                        results[fun]['mean'][indicesTmp] = np.mean(resultsPerm[permId][fun])
                        results[fun]['std'][indicesTmp] = np.std(resultsPerm[permId][fun])
                    except IndexError:
                        print 'Error: %s\n%s'%(str(indicesList), str(currentSet))
            
################################################################################
        else: 
            paramName = foldies[0].split(' ')[0]
            for folder in foldies:
                paramValue = folder.split(' ')[1]
                # Test if the parameter value is part of the values defined by parameter
                
#                valueList = [str(value) for value in indexRef[paramName]['val2ind'].keys()]
                valueList = ['%s'%parameters[paramName][3]%value for value in indexRef[paramName]['val2ind'].keys()]
                if paramValue not in valueList:
                    print "Parameter %s with value %s is not part of this optimization"%(str(paramName), str(paramValue))
                else:
                    currentSet[paramName] = float(paramValue)
                    try:
                        os.chdir(folder)
                    except:
                        print 'yo'
                    rec(results, currentSet, funs, parameters, indexRef)
                    os.chdir("..")
    
    def generateIndexRef(parameters):
        paramIndices = {}
        for param in parameters:
            paramIndices[param] = {}
            paramIndices[param]['val2ind'] = {}
            paramIndices[param]['ind2val'] = {}
            paramBounds = parameters[param]
            start = paramBounds[0]
            step = paramBounds[2]
            end = paramBounds[1]
            paramValues = np.arange(start, end + step / 2., step)
            if paramValues[-1] > end:
                paramValues = paramValues[:-1]
            i = 0
            for val in paramValues:
                # This little operation that appears stupid is needed to compare
                # two floats, to understand try 0.3 == 0.2 + 0.1
                precision = 100000000000000
                val = round(val*precision)/precision
                
                paramIndices[param]['val2ind'][val] = i
                paramIndices[param]['ind2val'][i] = val
                i+=1
            
        return paramIndices
    
    parameters = optInfos['parameters']
    nbInstances = optInfos['nbInstances']
    nbPermutations = optInfos['nbPermutations']
    
    # Get the number of dimensions
    dims = []
    for param in parameters:
        start = parameters[param][0]
        end = parameters[param][1]
        step = parameters[param][2]
        paramValues = np.arange(start, end + step / 2., step)
        if paramValues[-1] > end:
            paramValues = paramValues[:-1]
        nbSteps = len(paramValues)
        dims.append(nbSteps)
    
    # Add the number of permutations to the dimensions of the results to save 
    # each permutation
    dims.append(nbPermutations)
    
    # Initialize the matrices of results
    results = copy.deepcopy(funs)
    zeroMat = np.zeros(dims)
############ Change made to adapt to the results of the reservoir decoder#######
    for fun in results:
        results[fun] = {'mean': zeroMat.copy(),
                        'std': zeroMat.copy()}
        
################################################################################

    # Initialize the temporary variable currentSet to memorize the sets of parameters
    currentSet = copy.deepcopy(parameters)
    for param in currentSet:
        currentSet[param] = None
    
    # Create a dictionary with the indices for correspondence btw params dict
    # and results matrices
    paramsIndex = copy.deepcopy(currentSet)
    paramNames = parameters.keys()
    for i in range(len(paramNames)):
        paramsIndex[paramNames[i]] = i
    
    os.chdir(optFilesDir + "/data")
    indexRef = generateIndexRef(parameters)
    
    rec(results, currentSet, funs, parameters, indexRef)
    
    results['indexRef'] = indexRef
    results['paramsOrder'] = parameters.keys()
    
    return results

def getIndicesList(currentSet, parameters):
    outputList = []
    for param in parameters:
        index = int(round((currentSet[param]-parameters[param][0]) / parameters[param][2]))
        outputList.append(index)
    return outputList

def get_results_at_ts(resMat, timestep):
    # WARNING TASK SPECIFIC FUNCTION
    '''Get results at timestep
    Returns matrix of results
    '''
    def recurrentFun(dims, currentIndices):
        tmpDims = copy.deepcopy(dims)
        if len(tmpDims) == 0:
            newResMat[tuple(currentIndices)] = resMat[tuple(currentIndices + [timestep])]
        else:
            for i in range(tmpDims.pop(0)):
                currentIndices.append(i)
                recurrentFun(tmpDims, currentIndices)
                currentIndices.pop()
                
    dims = list(resMat.shape)
    dims.pop()
    currentIndices = []
    newResMat = np.zeros(dims)
    recurrentFun(dims, currentIndices)
    return newResMat

def get_best_results(resMat):
    # WARNING TASK SPECIFIC FUNCTION
    '''Return a matrix of the best performance value over all the timesteps'''
    def recurrentFun(dims, currentIndices):
        tmpDims = copy.deepcopy(dims)
        if len(tmpDims) == 0:
            newResMat[tuple(currentIndices)] = np.max(resMat[tuple(currentIndices)][200:900])
        else:
            for i in range(tmpDims.pop(0)):
                currentIndices.append(i)
                recurrentFun(tmpDims, currentIndices)
                currentIndices.pop()
    
    dims = list(resMat.shape)
    dims.pop()
    currentIndices = []
    newResMat = np.zeros(dims)
    recurrentFun(dims, currentIndices)
    return newResMat

def get_averaged_results(resMat, infBound=0, supBound=700):
    # WARNING TASK SPECIFIC FUNCTION
    '''Return a matrix of the averaged performance value over all the timesteps'''
    def recurrentFun(dims, currentIndices):
        tmpDims = copy.deepcopy(dims)
        if len(tmpDims) == 0:
            newResMat[tuple(currentIndices)] = np.mean(resMat[tuple(currentIndices)][infBound+200:supBound+200])
        else:
            for i in range(tmpDims.pop(0)):
                currentIndices.append(i)
                recurrentFun(tmpDims, currentIndices)
                currentIndices.pop()
                
    dims = list(resMat.shape)
    dims.pop()
    currentIndices = []
    newResMat = np.zeros(dims)
    recurrentFun(dims, currentIndices)
    return newResMat

def get_best_params(resultsMat):
    matTemp = resultsMat.copy()
    decreasingPerfIndices = []
    decreasingPerfs = []
    nbRes = 1
    dims = matTemp.shape
    for dim in dims:
        nbRes *= dim
    
    for i in range(nbRes):
        maxIndices = matTemp.argmax()
        maxIndices = np.unravel_index(maxIndices, dims)
        decreasingPerfIndices.append(maxIndices)
        decreasingPerfs.append(matTemp[maxIndices])
        matTemp[maxIndices] = 0
    
    return decreasingPerfIndices, decreasingPerfs

def retrieve_results(optFilesDir,
                     optimizationName,
                     getResFuncsDic):
    '''Retrieve the results in all the files and concatenate them in a file
    called concatResults.pk at the root of the optimization files
    
    Arguments:
    - optFilesDir: the root directory for all the optimization
    - optimizationName: the name given to the specific optimization you want to
        retrieve the results from
    - getResFuncsDic: stands for "get results functions dictionary".
    A dictionary that contains functions which get specific results from each
    result file.
    e.g.:
    The results file (like t0.0_iSp1.0_rad0.0_winS200_iSc1.6_wV0.0_nbN400_winC400_rS0.1__0.pk for example)
    contains the dictionary resDic with keys 'mean' and 'std', each key leading
    a float that represent respectively the mean performance and the standard
    deviation of performance. So my argument getResFuncsDic should be:
    
    def getMean(resDic):
        return resDic['mean']
        
    def getStd(resDic):
        return resDic['std']
        
    getResFuncsDic = {'mean': getMean,
                      'std': getStd}
    
    '''
    
    if optFilesDir[-1] != '/':
        optFilesDir += '/'
    fullPath = optFilesDir + optimizationName
    
    #Path of saved results
    savedResultsName = "concatResults.pk"
    
    # Load the parameters dictionary that contains the parameters explored
    f = open(fullPath + '/opt_infos.pk')
    optInfos = pk.load(f)
    f.close()
    
    # Show the parameters found in the result folder
    print "Parameters used for the exploration:"
    show_parameters(optFilesDir + optimizationName)
    
    # This function gathers the results from all the optimization files and
    # concatenate them inside a matrix
    resDic = concatenateResults(fullPath, optInfos, getResFuncsDic)
    
#    bestParamsList, bestResults = get_best_params(resDic['meanPerf']['mean'])
#    
#    print "\n" + str(get_val_with_indices(bestParamsList[0], resDic['paramsOrder'], resDic['indexRef']))
#    
#    # Save the recently processed data in a dictionary
#    resDic['bestParamsList'] = bestParamsList
#    resDic['bestResults'] = bestResults
    
    resDic['parameters'] = optInfos['parameters']
    
    f = open(optFilesDir+optimizationName+'/'+savedResultsName, 'w')
    pk.dump(resDic, f)
    f.close()

def retrieve_results_permutations(optFilesDir,
                                  optimizationName,
                                  getResFuncsDic):
    '''Retrieve the results in all the files and concatenate them in a file
    called concatResults.pk at the root of the optimization files
    
    Arguments:
    - optFilesDir: the root directory for all the optimization
    - optimizationName: the name given to the specific optimization you want to
        retrieve the results from
    - getResFuncsDic: stands for "get results functions dictionary".
    A dictionary that contains functions which get specific results from each
    result file.
    e.g.:
    The results file (like t0.0_iSp1.0_rad0.0_winS200_iSc1.6_wV0.0_nbN400_winC400_rS0.1__0.pk for example)
    contains the dictionary resDic with keys 'mean' and 'std', each key leading
    a float that represent respectively the mean performance and the standard
    deviation of performance. So my argument getResFuncsDic should be:
    
    def getMean(resDic):
        return resDic['mean']
        
    def getStd(resDic):
        return resDic['std']
        
    getResFuncsDic = {'mean': getMean,
                      'std': getStd}
    
    '''
    
    if optFilesDir[-1] != '/':
        optFilesDir += '/'
    fullPath = optFilesDir + optimizationName
    fullPathPerm = optFilesDir + optimizationName + '/permutations'
    
    # Load the parameters dictionary that contains the parameters explored
    f = open(fullPathPerm + '/opt_infos.pk')
    optInfos = pk.load(f)
    f.close()
    
    # Show the parameters found in the result folder
    print "Parameters used for the exploration:"
    show_parameters(fullPath)

    # This function gathers the results from all the optimization files and
    # concatenate them inside a matrix
    f = open(fullPath + '/concatResults.pk')
    resDic = pk.load(f)
    f.close()
    resDicPerm = concatenateResults_permutations(fullPathPerm, optInfos, getResFuncsDic)
    sigLevel = get_siglevel_allparams(resDic['mean perf']['mean'], resDicPerm['mean perf']['mean'])
    meanPermPerfs, stdPermPerfs = get_mean_std_permutations(resDicPerm['mean perf']['mean'])
    resDicPerm['all perms'] = {'mean': meanPermPerfs,
                               'std': stdPermPerfs}
    resDicPerm['significance level'] = sigLevel

    resDicPerm['parameters'] = optInfos['parameters']
    
    print "Significance level: %s"%str(sigLevel)
    
    f = open(fullPathPerm+'/concatResults.pk', 'w')
    pk.dump(resDicPerm, f)
    f.close()

def get_mean_std_permutations(permutationsPerfs):
    '''Return the mean and std for all the permutation performances
    '''
    paramsShape = list(permutationsPerfs.shape)
    paramsShape.pop(-1)
    nbParams = len(paramsShape)
    
    def recFunction(indices):
        if len(indices) == nbParams:
            meanPermPerfs[tuple(indices)] = np.mean(permutationsPerfs[tuple(indices)])
            stdPermPerfs[tuple(indices)] = np.std(permutationsPerfs[tuple(indices)])
        else:
            currentIndex = len(indices)
            nbDimPerParam = permutationsPerfs.shape[currentIndex]
            if nbDimPerParam > 1:
                for paramId in range(nbDimPerParam):
                    recFunction(indices + [paramId])
            else:
                recFunction(indices + [0])
    
    meanPermPerfs = np.zeros(paramsShape)
    stdPermPerfs = np.zeros(paramsShape)
    
    recFunction([])
        
    return meanPermPerfs, stdPermPerfs

def get_significance_level(decodingPerfs, permutationsPerfs):
    '''Return the significance level of a decoding as the probability that the
    performances are higher than the chance level
    
    This probability is computed as the number of permutations having higher
    performances than the original decoding performances plus 1 divided by the
    number of permutations
    
    Arguments:
    - decodingPerfs: a float for the real decoding performance
    - permutationsPerfs: a vector that contains all the performances values of
        each permutation for the same decoding data as decodingPerfs
    
    Returns:
    - sigLevel: the probability that the given performance is chance
    '''
    nbPermutations = len(permutationsPerfs)
    higherCount = 1
    for permPerf in permutationsPerfs:
        if decodingPerfs <= permPerf:
            higherCount += 1
    sigLevel = higherCount/nbPermutations
        
    return sigLevel

def get_siglevel_allparams(decodingPerfs, permutationsPerfs):
    '''Return the significance level of a decoding as the probability that the
    performances are higher than the chance level
    
    This probability is computed as the number of permutations having higher
    performances than the original decoding performances plus 1 divided by the
    number of permutations
    
    Arguments:
    - decodingPerfs: a vector that contain the performance values of the
        successive time steps
    - permutationsPerfs: a matrix that contains all the performances values of
        each permutation. First dimension is the time step and second is the
        different permutations
    
    Returns:
    - sigLevel: the probability that the given performance is above chance
    '''
    origShape = list(decodingPerfs.shape)
    permShape = list(permutationsPerfs.shape)
    permShape.pop(-1)
    if origShape != permShape:
        raise ValueError, 'decodingPerfs and permutationsPerfs must have the same dimension'
    
    def recFunction(indices):
        if len(indices) == len(decodingPerfs.shape):
            sigLevel[tuple(indices)] = get_significance_level(decodingPerfs[tuple(indices)],
                                                              permutationsPerfs[tuple(indices)])
        else:
            currentIndex = len(indices)
            nbDimPerParam = decodingPerfs.shape[currentIndex]
            if nbDimPerParam > 1:
                for paramId in range(nbDimPerParam):
                    recFunction(indices + [paramId])
            else:
                recFunction(indices + [0])
    
    sigLevel = np.zeros(decodingPerfs.shape)
    
    recFunction([])
        
    return sigLevel

################################################################################
################################################################################

#if __name__ == '__main__':
#    
###########################################################
#    optimizationName = '210111_block1_ca_da_windowParams_5th_verif'#################
###########################################################
#    
#    # Here you specify the root directory where are all the results
#    optFilesDir = "/media/data/IIT_collaboration/results/%s"%optimizationName
#    # WARNING: nothing else than the optimization files must be present in the
#    # root and subdirectories of the optimization files.
#    
#    # Path of saved results
#    savedResultsName = "concatResults.pk"
#    
#    # Load the parameters dictionary that contains the parameters explored
#    f = open(optFilesDir + '/opt_infos.pk')
#    optInfos = pk.load(f)
#    f.close()
#    
#    # Show the parameters found in the result folder
#    print "Parameters used for the exploration:"
#    show_parameters(optimizationName)
#    
#    # Here are functions that get the results from the results dictionary
#    def meanPerf(resDict):
#        return resDict['meanPerf']
#    
#    wantedDataFs = {'meanPerf':meanPerf}
#
#    # This function gathers the results from all the optimization files and
#    # concatenate them inside a matrix
#    resDic = concatenateResults(optFilesDir, optInfos, wantedDataFs)
#    
#    bestParamsList, bestResults = get_best_params(resDic['meanPerf']['mean'])
#    
#    print "\n" + str(get_val_with_indices(bestParamsList[0], resDic['paramsOrder'], resDic['indexRef']))
#    
#    # Save the recently processed data in a dictionary
#    resDic['bestParamsList'] = bestParamsList
#    resDic['bestResults'] = bestResults
#    
#    resDic['parameters'] = optInfos['parameters']
#    
#    f = open(optFilesDir+'/'+savedResultsName, 'w')
#    pk.dump(resDic, f)
#    f.close()
