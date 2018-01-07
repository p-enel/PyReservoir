#-*-coding: utf-8-*-
'''
Created on 27 juil. 2013

@author: pierre
'''
from __future__ import division
from ReservoirClass import Reservoir, np

class ReservoirOger(Reservoir):
    '''A reservoir like neural network
    
    Allows you to train and execute with trained weights a reservoir
    
    Arguments:
    - inputWeights: the matrix of input connections, must be consistent with the
        dimension of the input fed to the network.
    - internalWeights: the matrix of internal connections. For useful activity
        the this matrix chould be sparse and have a spectral radius close to 1.
    - readoutSize: number of neurons in the readout layer. Must be consistent
        with the output given for training.
    - outputFunction: the transfer function of the neurons in the network.
        Default value np.tanh works well with reservoirs.
    - tau: the time constant of the neurons. The neurons are leaky integrators,
        but the default value 1 set the neurons to a non-leaky regime.
    - readoutWeights: optional. If the network has already been trained or to
        test the network with specific readout weights, you can specify it here.
    - scipySparse: especially for reservoirs with a big number of neurons
        (~ 1500, to be confirmed...) the use of scipy.sparse module accelerate
        the processing, this argument should then be set to True.
    '''
    
    def _weight_inputs(self, input_, output):
    
        return self.outputFunction(np.dot(self.inputWeights, input_) + np.dot(self.internalWeights, output))
    def _weight_inputs_sparse(self, input_, output):
        return self.outputFunction(self.inputWeights.dot(input_) + self.internalWeights.dot(output))
    
    def train(self,
              inputList,
              desiredOutputList,
              regression='ridge',
              ridgeParam=0.0001,
              washout=0,
              subsetNodes=None,
              bias=True):
        ''' Train the network by computing the readout weights
        
        Arguments:
        - inputList: a list of matrices. The successive inputs fed to the
            network for training.
        - desiredOutputList: a list of matrices. The corresponding desired
            outputs of the inputs given in inputList argument.
        - regression: the type of regression learning used to compute the
            readout weights. Available options:
            ['ridge', 'linear'] so far
        - ridgeParam: if ridge regression is used, here is the corresponding
            parameter
        - washout: ...
        - subsetNodes: a list with the indices of the reservoir nodes that are
            used for the regression and thus connected to the readout
        - bias: the regression uses or not a bias -> the 'b' in 'y = a*x + b'
        '''
        
        if subsetNodes is None:
            subsetNodes = range(self.nbNeurons)
        else:
            for nodeId in subsetNodes:
                if nodeId >= self.nbNeurons or nodeId < 0:
                    raise ValueError
        
        # To fill the results matrix once and for all instead of concatenating at the end -> should win some time
        allInputsSize = 0
        maxInputSize = 0
        minInputSize = 9999999999
        for input_ in inputList:
            if input_.shape[0] > maxInputSize:
                maxInputSize = input_.shape[0]
            if input_.shape[0] < minInputSize:
                minInputSize = input_.shape[0]
            allInputsSize += input_.shape[0] - washout
        
        if washout > minInputSize:
            raise ValueError, 'The washout argument must be smaller than the smallest length of the inputs'
        
        resActivityTmp = np.zeros((maxInputSize, self.nbNeurons))
        
        if bias:
            resActivity = np.zeros((allInputsSize, self.nbNeurons + 1))
            resActivity[:,-1] = 1
            subsetNodes.append(self.nbNeurons)
            self.bias = True
        else:
            resActivity = np.zeros((allInputsSize, self.nbNeurons))
            self.bias = False
        
        outputIndex = 0
        self.readoutSize = desiredOutputList[0].shape[1]
        
#        print washout
        # Core of the training method -> optimization must be done here!
        for input_ in inputList:
            
            nbTimeSteps = input_.shape[0]
#            resActivityTmp = np.zeros((nbTimeSteps, self.nbNeurons))
            outputTmp = self.output
            for timeStep in xrange(nbTimeSteps):
                weightedInputs = self.weight_inputs(input_[timeStep], outputTmp)
                outputTmp = (1-1./self.tau) * outputTmp + (1./self.tau) * weightedInputs
                resActivityTmp[timeStep] = outputTmp
            resActivity[outputIndex:outputIndex+nbTimeSteps-washout, 0 : self.nbNeurons] = resActivityTmp[washout:nbTimeSteps]
            outputIndex += nbTimeSteps - washout
            resActivityTmp = np.zeros((maxInputSize, self.nbNeurons))
        
        # Reset the activity after training?
#        self.reset()
        
        # Concatenating desired output
        washoutDesOutput = np.zeros((allInputsSize, self.readoutSize))
        outputIndex = 0
        try:
            for i, desOutTrain in enumerate(desiredOutputList):
                nbTimeSteps = desOutTrain.shape[0]
                washoutDesOutput[outputIndex:outputIndex+nbTimeSteps-washout] = desOutTrain[washout:nbTimeSteps]
                outputIndex += nbTimeSteps - washout
        except:
            print 'desOutTrain index %d'%i
            print 'nbTimeSteps %d'%nbTimeSteps
            print 'outputIndex %d'%outputIndex
            print 'washoutDesOutput.shape %s'%str(washoutDesOutput.shape)
            print 'washout %d'%washout
            msg = 'There is an issue concatenating the desired output, this may be caused by several reasons:\n'
            msg += '- the number of time steps in an element of desiredOutputList is not consistent with its'
            msg += ' corresponding element in inputList\n'
            msg += '- readout size specified when creating the reservoir is not consistent with the size of'
            msg += ' a desire output element'
            raise Exception, msg
        
        resActivity = resActivity[:,subsetNodes]
        
        if regression == 'linear':
            inv_xTx = np.linalg.pinv(np.dot(resActivity.T, resActivity))

#                self.readoutWeights = np.linalg.solve(np.dot(resActivity.T, resActivity) + ridgeParam*np.identity(self.nbNeurons+bias), np.dot(resActivity.T,washoutDesOutput))
        elif regression == 'ridge':
            I = np.identity(len(subsetNodes))
            I[-1, -1] = 0 #No constraint on the fake neuron that actually is the bias
            try:
                inv_xTx = np.linalg.pinv(np.dot(resActivity.T, resActivity) + ridgeParam*I)
            except Exception, e:
                print 'resActivity ' + str(resActivity)
                print 'nbNeurons ' + str(self.nbNeurons)
                print 'subsetNodes ' + str(subsetNodes)
                print 'inputWeights ' + str(self.inputWeights)
                print 'internalWeights ' + str(self.internalWeights)
                print 'np.dot(resActivity.T, resActivity) ' + str(np.dot(resActivity.T, resActivity))
                raise e
                
#                self.readoutWeights = np.linalg.solve(np.dot(resActivity.T, resActivity), np.dot(resActivity.T,washoutDesOutput))
        xTy = np.dot(resActivity.T, washoutDesOutput)
        
        self.readoutWeights = np.dot(inv_xTx,xTy)
        if len(subsetNodes)-bias != self.nbNeurons:
            for nodeId in range(self.nbNeurons):
                if nodeId not in subsetNodes:
                    self.readoutWeights = np.insert(self.readoutWeights,nodeId,0,axis=0)
#            inv_xTx = np.linalg.pinv(np.dot(resActivity.T, resActivity) + ridgeParam*np.identity(self.nbNeurons+bias))
#            xTy = np.dot(resActivity.T, washoutDesOutput)
#            print 'xTx'
#            print np.dot(resActivity.T, resActivity)[:5,:5]
#            print 'resActivity'
#            print resActivity[:5,:5]
#            self.readoutWeights = np.dot(inv_xTx,xTy)
    
    def train_with_error(self,
                         inputList,
                         desiredOutputList,
                         regression='ridge',
                         ridgeParam=0.0001,
                         learningError=0.05,
                         washout=0,
                         subsetNodes=None,
                         bias=True):
        ''' Train the network by computing the readout weights
        
        Arguments:
        - inputList: a list of matrices. The successive inputs fed to the
            network for training.
        - desiredOutputList: a list of matrices. The corresponding desired
            outputs of the inputs given in inputList argument.
        - regression: the type of regression learning used to compute the
            readout weights. Available options:
            ['ridge', 'linear'] so far
        - ridgeParam: if ridge regression is used, here is the corresponding
            parameter
        - washout: ...
        - subsetNodes: a list with the indices of the reservoir nodes that are
            used for the regression and thus connected to the readout
        - bias: the regression uses or not a bias -> the 'b' in 'y = a*x + b'
        '''
        
        if subsetNodes is None:
            subsetNodes = range(self.nbNeurons)
        else:
            for nodeId in subsetNodes:
                if nodeId >= self.nbNeurons or nodeId < 0:
                    raise ValueError
        
        # To fill the results matrix once and for all instead of concatenating at the end -> should win some time
        allInputsSize = 0
        maxInputSize = 0
        minInputSize = 9999999999
        for input_ in inputList:
            if input_.shape[0] > maxInputSize:
                maxInputSize = input_.shape[0]
            if input_.shape[0] < minInputSize:
                minInputSize = input_.shape[0]
            allInputsSize += input_.shape[0] - washout
        
        if washout > minInputSize:
            raise ValueError, 'The washout argument must be smaller than the smallest length of the inputs'
        
        resActivityTmp = np.zeros((maxInputSize, self.nbNeurons))
        
        if bias:
            resActivity = np.zeros((allInputsSize, self.nbNeurons + 1))
            resActivity[:,-1] = 1
            subsetNodes.append(self.nbNeurons)
            self.bias = True
        else:
            resActivity = np.zeros((allInputsSize, self.nbNeurons))
            self.bias = False
        
        outputIndex = 0
            
        self.readoutSize = desiredOutputList[0].shape[1]
        
#        print washout
        # Core of the training method -> optimization must be done here!
        for input_ in inputList:
            
            nbTimeSteps = input_.shape[0]
#            resActivityTmp = np.zeros((nbTimeSteps, self.nbNeurons))
            outputTmp = self.output
            for timeStep in xrange(nbTimeSteps):
                weightedInputs = self.weight_inputs(input_[timeStep], outputTmp)
                outputTmp = (1-1./self.tau) * outputTmp + (1./self.tau) * weightedInputs
                resActivityTmp[timeStep] = outputTmp
            resActivity[outputIndex:outputIndex+nbTimeSteps-washout, 0 : self.nbNeurons] = resActivityTmp[washout:nbTimeSteps]
            outputIndex += nbTimeSteps - washout
            resActivityTmp = np.zeros((maxInputSize, self.nbNeurons))
        
        # Reset the activity after training?
#        self.reset()
        
        # Concatenating desired output
        washoutDesOutput = np.zeros((allInputsSize, self.readoutSize))
        outputIndex = 0
        try:
            for i, desOutTrain in enumerate(desiredOutputList):
                nbTimeSteps = desOutTrain.shape[0]
                washoutDesOutput[outputIndex:outputIndex+nbTimeSteps-washout] = desOutTrain[washout:nbTimeSteps]
                outputIndex += nbTimeSteps - washout
        except:
            print 'desOutTrain index %d'%i
            print 'nbTimeSteps %d'%nbTimeSteps
            print 'outputIndex %d'%outputIndex
            print 'washoutDesOutput.shape %s'%str(washoutDesOutput.shape)
            print 'washout %d'%washout
            msg = 'There is an issue concatenating the desired output, this may be caused by several reasons:\n'
            msg += '- the number of time steps in an element of desiredOutputList is not consistent with its'
            msg += ' corresponding element in inputList\n'
            msg += '- readout size specified when creating the reservoir is not consistent with the size of'
            msg += ' a desire output element'
            raise Exception, msg
        
        resActivity = resActivity[:,subsetNodes]
        
        if regression == 'linear':
            inv_xTx = np.linalg.pinv(np.dot(resActivity.T, resActivity))

#                self.readoutWeights = np.linalg.solve(np.dot(resActivity.T, resActivity) + ridgeParam*np.identity(self.nbNeurons+bias), np.dot(resActivity.T,washoutDesOutput))
        elif regression == 'ridge':
            I = np.identity(len(subsetNodes))
            I[-1, -1] = 0 #No constraint on the fake neuron that actually is the bias
            try:
                inv_xTx = np.linalg.pinv(np.dot(resActivity.T, resActivity) + ridgeParam*I)
            except Exception, e:
                print 'resActivity ' + str(resActivity)
                print 'nbNeurons ' + str(self.nbNeurons)
                print 'subsetNodes ' + str(subsetNodes)
                print 'wIn ' + str(self.inputWeights)
                print 'w ' + str(self.internalWeights)
                print 'np.dot(resActivity.T, resActivity) ' + str(np.dot(resActivity.T, resActivity))
                raise e
                
#                self.readoutWeights = np.linalg.solve(np.dot(resActivity.T, resActivity), np.dot(resActivity.T,washoutDesOutput))
        xTy = np.dot(resActivity.T, washoutDesOutput)
        
        newPerfectWeigths = np.dot(inv_xTx,xTy)
        deltaWeights = newPerfectWeigths - self.readoutWeights
        self.readoutWeights = self.readoutWeights + deltaWeights * (1-learningError)
        
        if len(subsetNodes)-bias != self.nbNeurons:
            for nodeId in range(self.nbNeurons):
                if nodeId not in subsetNodes:
                    self.readoutWeights = np.insert(self.readoutWeights,nodeId,0,axis=0)
#            inv_xTx = np.linalg.pinv(np.dot(resActivity.T, resActivity) + ridgeParam*np.identity(self.nbNeurons+bias))
#            xTy = np.dot(resActivity.T, washoutDesOutput)
#            print 'xTx'
#            print np.dot(resActivity.T, resActivity)[:5,:5]
#            print 'resActivity'
#            print resActivity[:5,:5]
#            self.readoutWeights = np.dot(inv_xTx,xTy)
        return resActivity
    
    def computeOutput(self):
        pass
    
    def computeReadout(self):
        '''Compute readout from output with readout weights'''
        if self.bias:
            newOutput = np.concatenate((self.output,[1]))
        else:
            newOutput = self.output
        self.readout = np.dot(newOutput, self.readoutWeights)
    
    def execute(self,
                input_):
        '''Feed the reservoir with the input and return the readout and
        reservoir output
        
        Argument:
        - input_: a single matrix.
        '''
        if self.readoutWeights is None:
            raise ValueError, 'You have to first train the network before executing it!'
        elif (self.readoutWeights.shape[0] != self.nbNeurons and self.readoutWeights.shape[0] != self.nbNeurons + 1)\
            or self.readoutWeights.shape[1] != self.readoutSize:
            raise ValueError, 'The size of the readout weights matrix is not consistent with the number of neurons (%d) or the readout size (%d)'%(self.nbNeurons, self.readoutSize)
        
        nbTimeSteps = input_.shape[0]
        
        if self.bias:
            resActivity = np.zeros((nbTimeSteps, self.nbNeurons + 1))
            resActivity[:,-1] = 1
        else:
            resActivity = np.zeros((nbTimeSteps, self.nbNeurons))
        
        outputTmp = self.output
        
        for timeStep in range(nbTimeSteps):
            weightedInputs = self.weight_inputs(input_[timeStep], outputTmp)
            outputTmp = (1-1/self.tau) * outputTmp + (1/self.tau) * weightedInputs
            resActivity[timeStep, 0 : self.nbNeurons] = outputTmp
        
        self.output = outputTmp
        
        readout = np.dot(resActivity, self.readoutWeights)
        
        return readout, resActivity[:, : self.nbNeurons]
    
    def execute_list(self,
                     inputList,
                     recordReservoir=False,
                     concatenated=False,
                     reset=True):
        '''Feed a list of inputs to the reservoir and record the readout activity
        
        Arguments:
        - inputList: the list of inputs fed to the reservoir...
        - recordReservoir: if True, returns also the activity of the reservoir
        - concatenated: if True, returns the activity of the readout concatenated
            if False, returns the activity of the readout in a list
        - reset: if False, does not reset the activity of the neurons inside the
            reservoir between each input in the list
        
        Returns:
        - resultsDic: a dictionary with the activity of the readout and the
            reservoir if asked, as list(s) (by defaut) or as matrix(/ces) if
            asked. See arguments...
        '''
        self.reset()
        
        if concatenated:
            nbTimeStepsTotal = 0
            for input_ in inputList:
                nbTimeStepsTotal += input_.shape[0]
    
            if recordReservoir:            
                resActivityAllInputs = np.zeros((nbTimeStepsTotal,self.nbNeurons))
            readoutActivityAllInputs = np.zeros((nbTimeStepsTotal,self.readoutSize))
        else:
            readoutActivityAllInputs = []
            if recordReservoir:
                resActivityAllInputs = []
        
        timeStepIndex = 0
        for input_ in inputList:
            readoutActivity, resActivity = self.execute(input_)
            
            if concatenated:
                nbTimeSteps = input_.shape[0]
                readoutActivityAllInputs[timeStepIndex:timeStepIndex+nbTimeSteps] = readoutActivity
                
                if recordReservoir:
                    resActivityAllInputs[timeStepIndex:timeStepIndex+nbTimeSteps] = resActivity
                    
                timeStepIndex += nbTimeSteps
            
            else:
                readoutActivityAllInputs.append(readoutActivity)
                
                if recordReservoir:
                    resActivityAllInputs.append(resActivity)
                
            if reset:
                self.reset()
        
        resultsDic = {'readoutActivity':readoutActivityAllInputs}
        if recordReservoir:
            resultsDic['reservoirActivity'] = resActivityAllInputs
            
        return resultsDic
    
