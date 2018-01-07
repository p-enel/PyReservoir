#-*-coding: utf-8-*-
'''
Created on 23 avr. 2012

@author: pierre
'''

import os

def build_path(path):
    
    if path[-1] == '/':
        path = path[:-1]
    
    head, tail = os.path.split(path)
    toBuild = [tail]
    
    while not os.path.lexists(head):
        head, tail = os.path.split(head)
        toBuild.append(tail)
    
    while len(toBuild) != 0:
        currentToBuild = toBuild.pop()
        head = os.path.join(head,currentToBuild)
        os.mkdir(head)
    
def organizeFiles(parameterSet, dataDirectory):
    try:
        os.chdir(dataDirectory)
    except OSError:
        raise ValueError, 'dataDirectory does not exist'
    
    fileName = ''
    fileDir = dataDirectory + '/'
    
#    lastName = parameterSet.keys()[-1]
    parameterSetTmp = parameterSet.copy()
#    last = parameterSetTmp.pop(lastName)
#    print 'List of the parameters optimized %s '%str(parameterSetTmp.keys())
    instance = parameterSetTmp.pop('inst')
    
    perm = False
    if 'permutationId' in parameterSet.keys():
        permutationId = parameterSet['permutationId']
        parameterSetTmp.pop('permutationId')
        perm = True
        
    parameterSetTmp.pop('seed')

    for paramName in parameterSetTmp:
        try:
            paramValue = parameterSetTmp[paramName][0]
        except:
            print 'p:' + str(parameterSetTmp)
            print 'p[:' + str(parameterSetTmp[paramName])
            raise Exception, 'Error in organize files...!'
        paramFormat = parameterSetTmp[paramName][1]
        paramLetter = parameterSetTmp[paramName][2]
        fileName += paramLetter + paramFormat % paramValue + '_'
        diry = '%s %s' %(paramName, paramFormat%paramValue)
        fileDir += diry + '/'
        try:
            os.chdir(diry)
        except OSError:
            os.mkdir(diry)
            os.chdir(diry)
    
#    paramValue = last[0]
#    paramFormat = last[1]
#    paramLetter = last[2]
#    
#    fileName += paramLetter + paramFormat % paramValue
    
    if perm:
        permutation = 'perm' + str(permutationId) + '_'
    else:
        permutation = ''
    
    fullPath = fileDir + fileName + permutation + str(instance) + '.pk'
    
    return fullPath

def set_optimization_dir(optDir):
    '''Enter the optimization root directory and create it if absent'''
    if not os.path.lexists(optDir):
        build_path(optDir)
        print "Folder %s missing, now created"%optDir
    
    dataFolder = optDir+'/data'
    if not os.path.lexists(dataFolder):
        build_path(dataFolder)
    
    os.chdir(dataFolder)
    
    return dataFolder
    
#    try:
#        os.chdir(optDir)
#    except OSError:
#        os.mkdir(optDir)
#        os.chdir(optDir)
#    
#    dataFolder = optDir+'/data'
#    try:
#        os.chdir(dataFolder)
#    except:
#        os.mkdir(dataFolder)
#        os.chdir(dataFolder)
#    
#    return os.getcwd()
