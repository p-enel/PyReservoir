'''
Created on 26 janv. 2011

@author: pier
'''
import numpy as np

def generateRandomWeights(nbUnitsIN=1,
                          nbUnitsOUT=None,
                          sparseness=1,
                          mask=None,
                          distribution='uniform',
                          distParams=[0, 1],
                          scaling=1,
                          spectralRadius=None,
                          seed=None):
    '''
    Distribution can be uniform or normal
    distParams: [mean, std] for normal distrib
                [min, max] for uniform
    '''
    if nbUnitsOUT is None:
        nbUnitsOUT = nbUnitsIN
    
    if spectralRadius is not None and nbUnitsIN != nbUnitsOUT:
        raise ValueError, 'spectralRadius can be defined only for square matrices (nbUnitsIN == nbUnitsOUT)'
    
    # Set the seed for the random weight generation
    np.random.seed(seed)
    
    # Uniform random distribution of weights:
    if distribution == 'uniform':
        if distParams is not None:
            minimum, maximum = distParams
        else:
            minimum, maximum = [0, 1]
        weightMatrix = (np.random.random((nbUnitsOUT,nbUnitsIN)) * (maximum - minimum) + minimum)
        
    # Normal (gaussian) random distribution of weights:
    elif distribution == 'normal':
        if distParams is not None:
            mu, sigma = distParams
        else:
            mu, sigma = [0, 1]
        weightMatrix = np.random.randn(nbUnitsOUT,nbUnitsIN) * sigma + mu
    
    weightMatrix = weightMatrix * scaling
    weightMatrix = weightMatrix * (np.random.random((nbUnitsOUT,nbUnitsIN)) < sparseness)
    
    if mask is not None:
        weightMatrix = weightMatrix * mask
        
    if spectralRadius is not None:
        try:
            weightMatrix = weightMatrix / np.amax(np.absolute(np.linalg.eigvals(weightMatrix))) * spectralRadius
        except:
            print 'weightMatrix : %s'%str(weightMatrix)
            print 'np.linalg.eigvals(weightMatrix) ' + str(np.linalg.eigvals(weightMatrix))
            print 'np.amax(np.absolute(np.linalg.eigvals(weightMatrix))) ' + str(np.amax(np.absolute(np.linalg.eigvals(weightMatrix))))
            print 'spectralRadius ' + spectralRadius
    
    return weightMatrix