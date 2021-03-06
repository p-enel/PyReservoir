#-*-coding: utf-8-*-
'''
Created on 6 sept. 2012

@author: pierre
'''
# Standard imports
from __future__ import division
import multiprocessing
import sys
import traceback
import os
import pickle as pk
from organize_files import organizeFiles, set_optimization_dir
from parameter_sets import generateUniqueSets, generateSets
import copy

import random
from random import shuffle

from ownUtils import gzip_save, gzip_load

__all__ = ['get_permutation_data', 'gridsearch_with_permutations']


def get_nbTrials_per_class(data):
    pass


def gen_fake_trialsPerClassOLD(nbTrials, classes):
    '''Generate fake trialsPerClass for generate_permutations method

    Arguments:
    - nbTrialsPerClass: {'className'} nbTrials

    Return:
    - fakeTrialsPerClass: {'className'} [nbTrials] int
    '''
    fakeTrialsPerClass = dict.fromkeys(classes)
    for class_ in classes:
        fakeTrialsPerClass[class_] = range(nbTrials)

    return fakeTrialsPerClass


def gen_fake_trialsPerClass(nbTrialsPerClass):
    '''Generate fake trialsPerClass for generate_permutations method

    Arguments:
    - nbTrialsPerClass: {'className'} nbTrials

    Return:
    - fakeTrialsPerClass: {'className'} [nbTrials] int
    '''
    fakeTrialsPerClass = dict.fromkeys(nbTrialsPerClass.keys())
    for class_ in nbTrialsPerClass:
        fakeTrialsPerClass[class_] = range(nbTrialsPerClass[class_])

    return fakeTrialsPerClass


def generate_permutations(trialsPerClass, seed=None):
    '''An iterator that generate permutations

    Arguments:
    - trialsPerClass: the decoding data organized by class in a dictionary.
        {'className'}[nbTrials]<np.ndarray>
        It does not have to be the actual data:
        {'className'}[nbTrials] int
        It will work with that king of structure
    - seed: seed for the permutations

    Return:
    - An iterator that returns successive permutations of the form:
        {'className'}[nbTrials]('className', trialId)
    '''
    classes = trialsPerClass.keys()

    nbTrialsPerClass = dict.fromkeys(trialsPerClass.keys())

    trialIds = []
    for class_ in classes:
        nbTrialsPerClass[class_] = len(trialsPerClass[class_])
        trialIds += zip([class_] * nbTrialsPerClass[class_],
                        range(nbTrialsPerClass[class_]))

    del trialsPerClass

    random.seed(seed)
    previousPerms = [trialIds[:]]

    while True:
        trialIdsTmp = trialIds[:]
        sameAsOld = True
        i = 0
        while sameAsOld and i < 100000:
            shuffle(trialIdsTmp)
            if trialIdsTmp not in previousPerms:
                sameAsOld = False
            i += 1

        newTrials = {}
        for class_ in classes:
            newTrials[class_] = trialIdsTmp[:nbTrialsPerClass[class_]]
            trialIdsTmp[:nbTrialsPerClass[class_]] = []

        yield newTrials


def get_permutation_data(permutation, dataByClasses):
    '''Apply a permutation on the decoding data

    Arguments:
    - permutation: a permutation as returned by the generate_permutations
        iterator
    - dataByClass: the decoding data with its original classes in the form!
        {'className'}[nbTrials]<np.ndarray>

    Return:
    - permutationData: a decoding data structure of the form of dataByClass but
        with the permutation applied
    '''
    permutationData = dict.fromkeys(dataByClasses.keys())

    for newClass in permutation.iterkeys():
        permutationData[newClass] = []
        for newTrialId in permutation[newClass]:
            oldClass, oldTrialId = newTrialId
            permutationData[newClass].append(
                dataByClasses[oldClass][oldTrialId])

    return permutationData


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
                    msg += '%s %d, ' % (param, parameterSet[param])
                else:
                    paramValue, paramFormat = parameterSet[param][:2]
                    msg += ('%s %s, ' % (param, paramFormat)) % paramValue

            if functionJob.permutations is not None:
                msg += '%d permutation' % functionJob.permutations
            else:
                msg = msg[:-2]

            print msg

            results = functionJob.resFunc(parameterSet,
                                          resFuncArgs,
                                          permutations=functionJob.permutations)

            f = open(fullPath, 'w')
            pk.dump(results, f)
            f.close()
        else:
            print 'Parameter set already done'

    except Exception:
        try:
            os.system('notify-send "Error in gridsearch %s!!!" -t 0' %
                      functionJob.optimizationName)
            os.system('play -q ~/Musique/sad_tombone.wav')
        except:
            pass
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print '#############################################################################################'
        print 'Error in parameter set:'
        print parameterSet
        print 'Traceback of exception: %s of %s' % (exc_value, str(exc_type))
        traceback.print_tb(exc_traceback, limit=10, file=sys.stdout)
        print '\n'
        raise exc_type, exc_value


def generateSets_with_permutations(parameters,
                                   nbPermutations=1,
                                   nbInstances=1,
                                   seed=None):

    uniqueSets = generateUniqueSets(parameters)

    finalSet = []
    if seed is not None:
        random.seed(seed)

    seeds = [int(random.random() * 10**12) for i in range(nbInstances)]

    for permId in range(nbPermutations):
        for instId in range(nbInstances):
            tmpSets = copy.deepcopy(uniqueSets)
            for paramSet in tmpSets:
                paramSet['inst'] = instId
                paramSet['permutationId'] = permId
                paramSet['seed'] = seeds[instId]
            finalSet += tmpSets

    random.shuffle(finalSet)
    return finalSet


def gridsearch_with_permutations(parameters,
                                 pathToResultsDir,
                                 optimizationName,
                                 nbInstances=5,
                                 nbProcessorUsed=1,
                                 testingMode=False,
                                 reservoirFunc=None,
                                 resFuncArgs={},
                                 nbPermutations=None,
                                 nbTrialsPerClass=None,
                                 #                                 nbTrialsTraining=None,
                                 #                                 nbTrialsValidation=None,
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
    if pathToResultsDir[-1] != '/':
        pathToResultsDir += '/'

    if nbPermutations is not None:
        assert isinstance(nbPermutations, int)
        optimizationRootDirectory = pathToResultsDir + \
            optimizationName + '/permutations/'
    else:
        optimizationRootDirectory = pathToResultsDir + optimizationName

    optimizationDataDirectory = set_optimization_dir(optimizationRootDirectory)

#   Permutations for recombined data in training only... check arguments as well
#    Cannot work anymore, the "gen_fake_trialsPerClass" function as been modified
################################################################################
#    if not os.path.isfile(optimizationRootDirectory+'permutations.pk'):
#        permutationsTraining = []
#        permutationsValidation = []
#        permutations = {'training': permutationsTraining,
#                        'validation': permutationsValidation}
#        fakeTrialsPerClassTraining = gen_fake_trialsPerClass(nbTrialsTraining, classes)
#        fakeTrialsPerClassValidation = gen_fake_trialsPerClass(nbTrialsValidation, classes)
#        permutatorTraining = generate_permutations(fakeTrialsPerClassTraining)
#        permutatorValidation = generate_permutations(fakeTrialsPerClassValidation)
#        for i in range(nbPermutations):
#            permutationsTraining.append(permutatorTraining.next())
#            permutationsValidation.append(permutatorValidation.next())
#        gzip_save(permutations, optimizationRootDirectory+'permutations.pk')
#    else:
#        permutations = gzip_load(optimizationRootDirectory+'permutations.pk')
#        try:
#            assert len(permutations['training']) == nbPermutations
#        except AssertionError:
#            msg = 'The number of permutations in the permutations.pk file does '
#            msg += 'not match the number of permutations passed as argument.'
#            raise ValueError, msg

    if nbPermutations is not None:
        if not os.path.isfile(optimizationRootDirectory + 'permutations.pk'):
            permutations = []
            fakeTrialsPerClass = gen_fake_trialsPerClass(nbTrialsPerClass)
            seedForPermutations = int(random.random() * 10**12)
            permutatorTraining = generate_permutations(
                fakeTrialsPerClass, seedForPermutations)
            for i in range(nbPermutations):
                permutations.append(permutatorTraining.next())
            gzip_save([permutations, seedForPermutations],
                      optimizationRootDirectory + 'permutations.pk')
        else:
            permutations, seedForPermutations = gzip_load(
                optimizationRootDirectory + 'permutations.pk')
            if len(permutations) < nbPermutations:
                permutations = []
                fakeTrialsPerClass = gen_fake_trialsPerClass(nbTrialsPerClass)
                permutatorTraining = generate_permutations(
                    fakeTrialsPerClass, seedForPermutations)
                for i in range(nbPermutations):
                    permutations.append(permutatorTraining.next())
                gzip_save([permutations, seedForPermutations],
                          optimizationRootDirectory + 'permutations.pk')
            elif len(permutations) > nbPermutations:
                permutations = permutations[:nbPermutations]
    else:
        permutations = None

    # Generate a list of the parameter sets explored
    if nbPermutations is not None:
        paramSets = generateSets_with_permutations(parameters,
                                                   nbPermutations=nbPermutations,
                                                   nbInstances=nbInstances,
                                                   seed=seed)
    else:
        paramSets = generateSets(parameters,
                                 nbInstances=nbInstances,
                                 seed=seed)

    nbParamSets = len(paramSets)
    print "Nb of parameters sets: %d" % nbParamSets

    # Optimization infos:
    optInfos = {'parameters': parameters,
                'nbInstances': nbInstances,
                'optimizationName': optimizationName}

    if nbPermutations is not None:
        optInfos['nbPermutations'] = nbPermutations

    # Saving the parameters variable for future processing
    f = open(optimizationRootDirectory + '/opt_infos.pk', 'w')
    pk.dump(optInfos, f)
    f.close()

    functionJobArguments = [(paramSets[i], resFuncArgs)
                            for i in range(nbParamSets)]

    functionJob.resFunc = reservoirFunc
    functionJob.optimizationName = optimizationName
    functionJob.dataDir = optimizationDataDirectory
    functionJob.permutations = permutations

    # Create a pool processes and launch the optimization on these processes
    if testingMode:
        for arguments in functionJobArguments:
            functionJob(arguments)
    else:
        pool = multiprocessing.Pool(processes=nbProcessorUsed)
        pool.map(functionJob, functionJobArguments)

    if notify:
        try:
            os.system('notify-send "Optimization %s is over!!!" -t 0' %
                      optimizationName)
        except:
            pass
