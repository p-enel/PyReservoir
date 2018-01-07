#-*-coding: utf-8-*-
'''
Created on 27 juil. 2013

@author: pierre
'''
from __future__ import division
from ReservoirClass import Reservoir, np

class ReservoirNSL(Reservoir):
    
    def __init__(self, *args, **kwargs):
        
        super(ReservoirNSL, self).__init__(*args, **kwargs)
        self.states = np.zeros(self.nbNeurons)
    
    def _weight_inputs(self, input_, output):
        return np.dot(self.inputWeights, input_) + np.dot(self.internalWeights, output)
    
    def _weight_inputs_sparse(self, input_, output):
        return self.inputWeights.dot(input_) + self.internalWeights.dot(output)
    
    def train(self,
              inputList,
              desiredOutputList,
              regression='ridge',
              ridgeParam=0.0001,
              intrinsicNoise=None,
              washout=0,
              subsetNodes=None,
              returnResActivity=False,
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
        - intrinsicNoise: gaussian noise added to the activity of the reservoir
            nodes. tuple - (mu, sigma, [<np.array nbTimeSteps>]). The values of
            noise for each time step is computed at each time step with the
            parameters mu and sigma and the seed corresponding to that time step
            of the corresponding input in the list
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
        
        if intrinsicNoise is not None:
            assert len(intrinsicNoise) == 3
            assert type(intrinsicNoise[2]) == list
        
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
        for inputId, input_ in enumerate(inputList):
            
            nbTimeSteps = input_.shape[0]
#            resActivityTmp = np.zeros((nbTimeSteps, self.nbNeurons))
            
            if intrinsicNoise is not None:
                intrinsicNoiseSeeds = intrinsicNoise[2][inputId]
                assert len(intrinsicNoiseSeeds) == input_.shape[0]
                
                for timeStep in xrange(nbTimeSteps):
                    weightedInputs = self.weight_inputs(input_[timeStep], self.output)
                    deltaStates =  self.tau_inv * (weightedInputs - self.states)
                    np.random.seed(intrinsicNoiseSeeds[timeStep])
                    self.states = self.states + deltaStates + np.random.normal(intrinsicNoise[0], intrinsicNoise[1], self.nbNeurons)
                    self.output = self.outputFunction(self.states)
                    resActivityTmp[timeStep] = self.output
            else:
                for timeStep in xrange(nbTimeSteps):
                    weightedInputs = self.weight_inputs(input_[timeStep], self.output)
                    deltaStates =  self.tau_inv * (weightedInputs - self.states)
                    self.states = self.states + deltaStates
                    self.output = self.outputFunction(self.states)
                    resActivityTmp[timeStep] = self.output
            
            resActivity[outputIndex:outputIndex+nbTimeSteps-washout, 0 : self.nbNeurons] = resActivityTmp[washout:nbTimeSteps]
            outputIndex += nbTimeSteps - washout
            resActivityTmp = np.zeros((maxInputSize, self.nbNeurons))
        
        # Reset the activity after training?
#        self.reset()
        
        # Concatenating desired output
        washoutDesOutput = np.zeros((allInputsSize, self.readoutSize))
        outputIndex = 0
        try:
            for i, desiredOutput in enumerate(desiredOutputList):
                nbTimeSteps = desiredOutput.shape[0]
                washoutDesOutput[outputIndex:outputIndex+nbTimeSteps-washout] = desiredOutput[washout:nbTimeSteps]
                outputIndex += nbTimeSteps - washout
        except:
            print 'desiredOutput index %d'%i
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
        
        print "Starting inversion of dot(resActivity.T, resActivity)"
        
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
        print "Inversion done"
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
        if returnResActivity:
            return resActivity
            
    def reset(self):
        '''Reset the values of states and output to zero'''
        super(ReservoirNSL, self).reset()
        self.set_state(0)
    
    def set_state(self, stateValue):
        '''Set the state of the reservoir neurons
        
        Arguments:
        - it can either be a single real value or a vector the size of the
            reservoir
        '''
        if type(stateValue) not in [float, int]:
            if type(stateValue) == np.ndarray and stateValue.shape == (self.nbNeurons,):
                self.states = stateValue
            else:
                raise ValueError, 'The argument for set_state method is not valid'
        else:
            self.states[:] = stateValue
    
    def computeOutput(self):
        '''Compute output from state with transfer function'''
        self.output = self.outputFunction(self.states)
    
    def computeReadout(self):
        '''Compute readout from output with readout weights'''
        if self.bias:
            newOutput = np.concatenate((self.output,[1]))
        else:
            newOutput = self.output
        self.readout = np.dot(newOutput, self.readoutWeights)
    
    def execute(self,
                input_,
                intrinsicNoise=None,
                returnStates=False):
        '''Feed the reservoir with the input and return the readout and
        reservoir output
        
        Argument:
        - input_: a single matrix.
        - intrinsicNoise: gaussian noise added to the activity of the reservoir
            nodes. tuple - (mu, sigma, <np.array nbTimeSteps>). The values of
            noise for each time step is computed at each time step with the
            parameters mu and sigma and the seed corresponding to that time step
        '''
        if self.readoutWeights is None:
            raise ValueError, 'You have to first train the network before executing it!'
        elif (self.readoutWeights.shape[0] != self.nbNeurons and self.readoutWeights.shape[0] != self.nbNeurons + 1)\
            or self.readoutWeights.shape[1] != self.readoutSize:
            raise ValueError, 'The size of the readout weights matrix is not consistent with the number of neurons (%d) or the readout size (%d)'%(self.nbNeurons, self.readoutSize)
        
        if intrinsicNoise != None:
            assert len(intrinsicNoise) == 3
            assert type(intrinsicNoise[0]) in [float, int] and type(intrinsicNoise[1]) in [float, int]
        
        nbTimeSteps = input_.shape[0]
        
        if self.bias:
            resActivity = np.zeros((nbTimeSteps, self.nbNeurons + 1))
            resActivity[:,-1] = 1
        else:
            resActivity = np.zeros((nbTimeSteps, self.nbNeurons))
        if returnStates:
            savedStates = np.zeros((nbTimeSteps, self.nbNeurons))
        
        statesTmp = self.states
        outputTmp = self.output
        
        if intrinsicNoise != None:
            intrinsicNoiseSeeds = intrinsicNoise[2]
            assert len(intrinsicNoiseSeeds) == nbTimeSteps
            for timeStep in range(nbTimeSteps):
                weightedInputs = self.weight_inputs(input_[timeStep], outputTmp)
                deltaStates =  (self.tau_inv) * (weightedInputs - statesTmp)
                np.random.seed(intrinsicNoiseSeeds[timeStep])
                statesTmp = statesTmp + deltaStates + np.random.normal(intrinsicNoise[0], intrinsicNoise[1], self.nbNeurons)
                outputTmp = self.outputFunction(statesTmp)
                if returnStates:
                    savedStates[timeStep] = statesTmp
                resActivity[timeStep, 0 : self.nbNeurons] = outputTmp
        else:        
            for timeStep in range(nbTimeSteps):
                weightedInputs = self.weight_inputs(input_[timeStep], outputTmp)
                deltaStates =  (self.tau_inv) * (weightedInputs - statesTmp)
                statesTmp = statesTmp + deltaStates
                outputTmp = self.outputFunction(statesTmp)
                if returnStates:
                    savedStates[timeStep] = statesTmp
                resActivity[timeStep, 0 : self.nbNeurons] = outputTmp
        
        readout = np.dot(resActivity, self.readoutWeights)
        
        self.states = statesTmp
        self.output = outputTmp
        self.readout = readout[-1,:]
        
        if returnStates:
            return readout, resActivity[:, : self.nbNeurons], savedStates
        else:
            return readout, resActivity[:, : self.nbNeurons]
    
    def execute_list(self,
                     inputList,
                     intrinsicNoise=None,
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
    
    def freerun(self,
                nbTimeSteps=1):
        '''Run the reservoir in free run mode
        
        Free run means that the reservoir takes its on readout output as inputs
        and run in a closed loop.
        
        Arguments:
        - nbTimeSteps: number of time steps that the reservoir will be in free run
        '''
        
        if self.readout is None:
            raise ValueError, "The execution of at least one time step is necessary before starting a freerun"
        
        if self.readoutWeights.shape[0] == self.nbNeurons + 1:
            resActivity = np.zeros((nbTimeSteps, self.nbNeurons + 1))
            resActivity[:,-1] = 1
        else:
            resActivity = np.zeros((nbTimeSteps, self.nbNeurons))
        
        readout = np.zeros((nbTimeSteps, self.readoutSize))
        output = np.zeros((nbTimeSteps, self.nbNeurons))
        
        statesTmp = self.states
        outputTmp = self.output
        readoutTmp = self.readout
        
        for timeStep in range(nbTimeSteps):
#            self.weightedInputs = np.dot(self.inputWeights, self.readout) + np.dot(self.internalWeights, self.output)
            weightedInputs = self.weight_inputs(readoutTmp, outputTmp)
            deltaStates =  (1. / self.tau) * (weightedInputs - statesTmp)
            statesTmp = statesTmp + deltaStates
            outputTmp = self.outputFunction(self.states)
            resActivity[timeStep, 0 : self.nbNeurons] = outputTmp
            readoutTmp[timeStep,:] = np.dot(self.readoutWeights, self.outputWithBias)
            
            readout[timeStep] = self.readout
            output[timeStep] = outputTmp
        
        self.states = statesTmp
        self.output = outputTmp
        self.readout = readoutTmp
        
        return readout, output
