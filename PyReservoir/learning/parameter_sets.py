#-*-coding: utf-8-*-
'''
Created on 23 avr. 2012

@author: pierre
'''
from __future__ import division
# Standard input
import numpy as np
import pickle as pk
import copy, random

def generateUniqueSets(d, currentSet={}):
    output = []
        
    if len(d) == 0:
        return [currentSet]
    else:
        localDict = d.copy()
        paramName = localDict.keys()[0]
        paramValues = localDict.pop(localDict.keys()[0])
        start = paramValues[0]
        step = paramValues[2]
        end = paramValues[1]
        format_ = paramValues[3]
        letter = paramValues[4]
        assert start <= end, "Optimization starting value of param %s is higher than ending value"%paramName
        paramValues = np.arange(start, end + step / 2., step)
        if paramValues[-1] > end:
            paramValues = paramValues[:-1]
        for param in paramValues:
            newSet = currentSet.copy()
            newSet[paramName] = [param, format_, letter]
            output = output + generateUniqueSets(localDict, newSet)
        return output

def generateSets(d, nbInstances=1, seed=None):
    uniqueSets = generateUniqueSets(d)
    finalSet = []
    if seed is not None:
        random.seed(seed)
    else:
        seeds = [int(random.random() * 10**12) for i in range(nbInstances*len(uniqueSets))]
    for i in range(nbInstances):
        tmpSets = copy.deepcopy(uniqueSets)
        for paramSet in tmpSets:
            paramSet['inst'] = i
            if seed is not None:
                paramSet['seed'] = seed
            else:
                paramSet['seed'] = seeds.pop()
        finalSet += tmpSets
    
    random.shuffle(finalSet)
    return finalSet

def show_parameters(optFilesDir):
    
    if optFilesDir[-1] != '/':
        optFilesDir += '/'
    f = open(optFilesDir+'opt_infos.pk','r')
    optInfos = pk.load(f)
    parameters = optInfos['parameters']
    
    for param in parameters:
        if parameters[param][0] == parameters[param][1]:
            print ('%-20s %s')%(param, parameters[param][3])%(parameters[param][0])
        else:
            print ('%-20s from %s to %s with step %s'%(param, parameters[param][3],parameters[param][3],parameters[param][3]))%(parameters[param][0],parameters[param][1],parameters[param][2])

def get_val_with_indices(indices, paramsOrder, indexRef):
    values = copy.deepcopy(indexRef)
    for i in range(len(paramsOrder)):
        param = paramsOrder[i]
        values[param] = indexRef[param]['ind2val'][indices[i]]
    return values

def get_values_from_indices(indices, resultsDic):
    paramsOrder = resultsDic['paramsOrder']
    indexRef = resultsDic['indexRef']
    return get_val_with_indices(indices, paramsOrder, indexRef)

def get_indices_from_values(values, resultsDic):
    indexRef = resultsDic['indexRef']
    paramSet = []
    for param in values:
        paramSet.append(indexRef[param]['val2ind'][values[param]])
    return tuple(paramSet)

def values_to_str(values):
    string = ''
    for param in values:
        string += param + ': %.2f | '%values[param]
    return string
