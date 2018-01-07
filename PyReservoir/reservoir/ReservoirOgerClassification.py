#-*-coding: utf-8-*-
'''
Created on 26 juil. 2013

@author: pierre
'''
from ReservoirOgerNodes import ReservoirOger, np

class ReservoirOgerClassification(ReservoirOger):
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
    
    def test(self,
             inputList,
             desiredOutputList,
             performanceFunc,
             initialState=None,
             perfFuncArgs=None):
        '''Present inputs to the reservoir and assess performances of the
        reservoir
        
        Arguments:
        - inputList: a list of matrices containing the input data to test
        - desiredOutputList: a list of matrices containing the desired output
            data to assess performances
        - performanceFunc: the function that assess the performance of the
            reservoir. This function takes two main inputs:
                - the reservoir activity elicited by the input(s)
                - the desired output class of the input(s)
        - perfFuncArgs: a dictionary containing the other arguments of the
            performance function
        '''
        nbInputs = len(inputList)
        readoutActivityList = []
        reservoirActivityList = []
        performances = []
        if initialState is not None:
            self.set_output(initialState)
        else:
            self.reset()
        for i in range(nbInputs):
            readoutActivity, reservoirActivity = self.execute(inputList[i])
            readoutActivityList.append(readoutActivity)
            reservoirActivityList.append(reservoirActivity)
            perfs = performanceFunc(readoutActivity,
                                    desiredOutputList[i],
                                    **perfFuncArgs)
            performances.append(perfs)
            if initialState is not None:
                self.set_output(initialState)
            else:
                self.reset()
        
        return performances, readoutActivityList, reservoirActivityList
#         return performances, readoutActivityList
    
    def test_classification(self,
                            inputList,
                            desiredOutputList,
                            performanceFunc,
                            initialState=0,
                            perfFuncArgs=None):
        '''Present inputs to the reservoir and assess performances of the
        reservoir
        
        Arguments:
        - inputList: a list of matrices containing the input data to test
        - desiredOutputList: a list of matrices containing the desired output
            data to assess performances
        - performanceFunc: the function that assess the performance of the
            reservoir. This function takes two main inputs:
                - the reservoir activity elicited by the input(s)
                - the desired output class of the input(s)
        - perfFuncArgs: a dictionary containing the other arguments of the
            performance function
        '''
        nbInputs = len(inputList)
        readoutActivityList = []
        reservoirActivityList = []
        performances = []
        choices = []
        self.set_output(initialState)
        self.computeOutput()
        for i in range(nbInputs):
            readoutActivity, reservoirActivity = self.execute(inputList[i])
            readoutActivityList.append(readoutActivity)
            reservoirActivityList.append(reservoirActivity)
            perfs, choice = performanceFunc(readoutActivity,
                                            desiredOutputList[i],
                                            **perfFuncArgs)
            performances.append(perfs)
            choices.append(choice)
            self.set_output(initialState)
            self.computeOutput()
        
        return performances, choices, readoutActivityList, reservoirActivityList
    
    def cross_validation(self,
                         inputByClass,
                         desiredOutByClass,
                         performanceFunc,
                         validationSize=0.2,
                         perfFuncArgs=None,
                         trainParams=None,
                         seed=None,
                         returnActivity=False):
        '''Just what it looks to be!
        
        Arguments:
        - inputByClass: a dictionary whose keys are the class names and the
            values are lists containing the different trial inputs in matrices:
            {'className'} [nbTrials] <np.ndarray nbTimeSteps * nbInputs>
        - performanceFunc: the function that compute the performances of each
            output TO BE COMPLETED
        - validationSize: a float between 0 and 1 for the proportion of the set
            that is used for cross-validation
        - perfFuncArgs: additional arguments passed to the performanceFunc
        - trainParams: additional arguments passed to the train method
        - seed: NOT USED YET
        - returnActivity: if set to True, return all the data relative to each
            test in each fold in lists: test inputs, reservoir activity in test,
            readout activity in test, desired output in test.
        
        Returns:
        - perfsAllFold: performance for each fold test in the cross-validation
        Additional arguments if the returnActivity switch is activated:
        - testInputAllFold: the inputs of each fold to test the reservoir
        - reservoirAllFold: reservoir activity of each fold to test the reservoir
        - readoutAllFold: readout activity of each fold to test the reservoir
        - desOutputAllFold: desired output of each fold to test the reservoir
        '''
        nbTrials = len(inputByClass.values()[0])
        
        if True in [len(trials) != nbTrials for trials in inputByClass.values()]:
            msg = 'There must be the same number of trials for each class'
            raise ValueError, msg
        
        nbFold = int(np.round(1/validationSize))
        nbValidTrials = int(np.ceil(validationSize * nbTrials))
        nbTrainingTrials = nbTrials - nbValidTrials
        perfsAllFold = []
        desOutClassAllFold = []
        
        if returnActivity:
            testInputAllFold = []
            readoutAllFold = []
            reservoirAllFold = []
            desOutputAllFold = []
        
        for fold in range(nbFold):
            trainInputs = []
            validnInputs = []
            trainDesOut = []
            validnDesOut = []
            desOutputClass = []

            for class_ in inputByClass:
                trainInputs += inputByClass[class_][:fold*nbValidTrials]
                trainInputs += inputByClass[class_][(fold+1)*nbValidTrials:]
                trainDesOut += [desiredOutByClass[class_]] * nbTrainingTrials
                validnInputs += inputByClass[class_][fold*nbValidTrials:(fold+1)*nbValidTrials]
                validnDesOut += [desiredOutByClass[class_]] * nbValidTrials
                desOutputClass += [class_] * nbValidTrials
            
            self.train(trainInputs,
                       trainDesOut,
                       **trainParams)
            
            perfs, readoutActivity, resActivity = self.test(validnInputs,
                                                            validnDesOut,
                                                            performanceFunc,
                                                            perfFuncArgs=perfFuncArgs)
                
            perfsAllFold.append(np.mean(perfs))
            desOutClassAllFold += desOutputClass
            if returnActivity:
                testInputAllFold += validnInputs
                readoutAllFold += readoutActivity
                reservoirAllFold += resActivity
                desOutputAllFold += validnDesOut
        
        
        if returnActivity:
            return perfsAllFold, desOutClassAllFold, testInputAllFold, reservoirAllFold, readoutAllFold, desOutputAllFold
        else:
            return perfsAllFold, desOutClassAllFold
    
    def cross_validation_02(self,
                            inputByClass,
                            desiredOutByClass,
                            performanceFunc,
                            validationSize=0.2,
                            perfFuncArgs=None,
                            initialState=None,
                            trainParams=None,
                            trainingPerfs=False,
                            seed=None,
                            returnActivity=False):
        '''Just what it looks to be!
        
        Arguments:
        - inputByClass: a dictionary whose keys are the class names and the
            values are lists containing the different trial inputs in matrices:
            {'className'} [nbTrials] <np.ndarray nbTimeSteps * nbInputs>
        - performanceFunc: the function that compute the performances of each
            output TO BE COMPLETED
        - validationSize: a float between 0 and 1 for the proportion of the set
            that is used for cross-validation
        - perfFuncArgs: additional arguments passed to the performanceFunc
        - trainParams: additional arguments passed to the train method
        - trainingPerfs: boolean to compute and return or not the training
            performances
        - seed: NOT USED YET
        - returnActivity: if set to True, return all the data relative to each
            test in each fold in lists: test inputs, reservoir activity in test,
            readout activity in test, desired output in test.
        
        Returns:
        - perfsAllFold: performance for each fold test in the cross-validation
        Additional arguments if the returnActivity switch is activated:
        - testInputAllFold: the inputs of each fold to test the reservoir
        - reservoirAllFold: reservoir activity of each fold to test the reservoir
        - readoutAllFold: readout activity of each fold to test the reservoir
        - desOutputAllFold: desired output of each fold to test the reservoir
        '''
        nbTrials = len(inputByClass.values()[0])
        
        if True in [len(trials) != nbTrials for trials in inputByClass.values()]:
            msg = 'There must be the same number of trials for each class'
            raise ValueError, msg
        
        nbValidTrials = int(np.ceil(validationSize * nbTrials))
        nbFold = int(np.ceil(nbTrials/nbValidTrials))
        nbTrainingTrials = nbTrials - nbValidTrials
        perfsAllFold = []
        desOutClassAllFold = []
        
        if returnActivity:
            testInputAllFold = []
            readoutAllFold = []
            reservoirAllFold = []
            desOutputAllFold = []
        if trainingPerfs:
            perfsTrainingAllFold = []
            desOutClassTrainingAllFold = []
            if returnActivity:
                testInputTrainingAllFold = []
                readoutTrainingAllFold = []
                reservoirTrainingAllFold = []
                desOutputTrainingAllFold = []
        
        for fold in range(nbFold):
            if initialState is not None:
                self.set_state(initialState)
            trainInputs = []
            validnInputs = []
            trainDesOut = []
            validnDesOut = []
            desOutputClass = []
            trainDesOutputClass = []
            
            for class_ in inputByClass:
                trainInputsTmp = inputByClass[class_][:fold*nbValidTrials]
                trainInputsTmp += inputByClass[class_][(fold+1)*nbValidTrials:]
                nbTrainingTrials = len(trainInputsTmp)
                nbValidTrialsTmp = nbTrials - nbTrainingTrials
                trainInputs += trainInputsTmp
                trainDesOut += [desiredOutByClass[class_]] * nbTrainingTrials
                trainDesOutputClass += [class_] * nbTrainingTrials
                validnInputs += inputByClass[class_][fold*nbValidTrials:(fold+1)*nbValidTrials]
                validnDesOut += [desiredOutByClass[class_]] * nbValidTrialsTmp
                desOutputClass += [class_] * nbValidTrialsTmp
            
            self.train(trainInputs,
                       trainDesOut,
                       **trainParams)
            
            if trainingPerfs:
                (perfsTraining,
                 readoutActivityTraining,
                 resActivityTraining) = self.test(trainInputs,
                                                  trainDesOut,
                                                  performanceFunc,
                                                  initialState=initialState,
                                                  perfFuncArgs=perfFuncArgs)
            
            (perfs,
             readoutActivity,
             resActivity) = self.test(validnInputs,
                                      validnDesOut,
                                      performanceFunc,
                                      initialState=initialState,
                                      perfFuncArgs=perfFuncArgs)
                
            perfsAllFold.append(perfs)
            desOutClassAllFold.append(desOutputClass)
            if returnActivity:
                testInputAllFold.append(validnInputs)
                readoutAllFold.append(readoutActivity)
                reservoirAllFold.append(resActivity)
                desOutputAllFold.append(validnDesOut)
            if trainingPerfs:
                perfsTrainingAllFold.append(np.mean(perfsTraining))
                desOutClassTrainingAllFold.append(trainDesOutputClass)
                if returnActivity:
                    testInputTrainingAllFold.append(trainInputs)
                    readoutTrainingAllFold.append(readoutActivityTraining)
                    reservoirTrainingAllFold.append(resActivityTraining)
                    desOutputTrainingAllFold.append(trainDesOut)
        
        testingResults = {'perfs': perfsAllFold,
                          'desired class': desOutClassAllFold}
        if returnActivity:
            testingResults.update({'inputs': testInputAllFold,
                                   'reservoir activity': reservoirAllFold,
                                   'readout activity': readoutAllFold,
                                   'desired output': desOutputAllFold})
        if trainingPerfs:
            trainingResults = {'perfs': perfsTrainingAllFold,
                               'desired class': desOutClassTrainingAllFold}
            if returnActivity:
                trainingResults.update({'inputs': testInputTrainingAllFold,
                                        'reservoir activity': reservoirTrainingAllFold,
                                        'readout activity': reservoirTrainingAllFold,
                                        'desired output': desOutputTrainingAllFold})
        
        if trainingPerfs:
            return trainingResults, testingResults
        else:
            return testingResults
    
    def cross_validation_classification(self,
                                        inputByClass,
                                        desiredOutByClass,
                                        readoutNeuronsClasses,
                                        performanceFunc,
                                        validationSize=0.2,
                                        perfFuncArgs=None,
                                        trainParams=None,
                                        trainingPerfs=False,
                                        initialState=0,
                                        seed=None,
                                        returnActivity=False):
        '''Just what it looks to be!
        
        Arguments:
        - inputByClass: a dictionary whose keys are the class names and the
            values are lists containing the different trial inputs in matrices:
            {'className'} [nbTrialsPerClass] <np.ndarray nbTimeSteps * nbInputs>
        - performanceFunc: the function that compute the performances of each
            output TO BE COMPLETED
        - validationSize: a float between 0 and 1 for the proportion of the set
            that is used for cross-validation
        - perfFuncArgs: additional arguments passed to the performanceFunc
        - trainParams: additional arguments passed to the train method
        - seed: NOT USED YET
        - returnActivity: if set to True, return all the data relative to each
            test in each fold in lists: test inputs, reservoir activity in test,
            readout activity in test, desired output in test.
        
        Returns:
        - perfsAllFold: performance for each fold test in the cross-validation
        Additional arguments if the returnActivity switch is activated:
        - testInputAllFold: the inputs of each fold to test the reservoir
        - reservoirAllFold: reservoir activity of each fold to test the reservoir
        - readoutAllFold: readout activity of each fold to test the reservoir
        - desOutputAllFold: desired output of each fold to test the reservoir
        '''
        nbTrials = len(inputByClass.values()[0])
        
        if True in [len(trials) != nbTrials for trials in inputByClass.values()]:
            msg = 'There must be the same number of trials for each class'
            raise ValueError, msg
        nbValidTrials = int(np.ceil(validationSize * nbTrials))
        nbFold = int(np.ceil(nbTrials/nbValidTrials))
        
        classes = inputByClass.keys()
        classes.sort()
        
        perfsAllFold = []
        desOutClassAllFold = []
        choicesAllFold = []
        if returnActivity:
            testInputAllFold = []
            readoutAllFold = []
            reservoirAllFold = []
            desOutputAllFold = []
        if trainingPerfs:
            perfsTrainingAllFold = []
            desOutClassTrainingAllFold = []
            choicesTrainingAllFold = []
            if returnActivity:
                testInputTrainingAllFold = []
                readoutTrainingAllFold = []
                reservoirTrainingAllFold = []
                desOutputTrainingAllFold = []
        
        for fold in range(nbFold):
            trainInputs = []
            trainInputsTmp = []
            validnInputs = []
            trainDesOut = []
            validnDesOut = []
            desOutputClass = []
            trainDesOutputClass = []

            for class_ in classes:
                trainInputsTmp = inputByClass[class_][:fold*nbValidTrials]
                trainInputsTmp += inputByClass[class_][(fold+1)*nbValidTrials:]
                nbTrainingTrials = len(trainInputsTmp)
                trainInputs += trainInputsTmp
                trainDesOut += [desiredOutByClass[class_]] * nbTrainingTrials
                trainDesOutputClass += [class_] * nbTrainingTrials
                validnInputs += inputByClass[class_][fold*nbValidTrials:(fold+1)*nbValidTrials]
                validnDesOut += [desiredOutByClass[class_]] * nbValidTrials
                desOutputClass += [class_] * nbValidTrials
            
            print 'fold : ' + str(fold)
            self.set_state(initialState)
            self.computeOutput()
            self.train(trainInputs,
                       trainDesOut,
                       **trainParams)
            
            if trainingPerfs:
                (perfsTraining,
                 choicesTraining,
                 readoutActivityTraining,
                 resActivityTraining) = self.test_classification(trainInputs,
                                                                 trainDesOut,
                                                                 performanceFunc,
                                                                 initialState=initialState,
                                                                 perfFuncArgs=perfFuncArgs)
                choicesTrainingClass = [readoutNeuronsClasses[choice] for choice in choicesTraining]
            
            (perfs,
             choices,
             readoutActivity,
             resActivity) = self.test_classification(validnInputs,
                                                     validnDesOut,
                                                     performanceFunc,
                                                     initialState=initialState,
                                                     perfFuncArgs=perfFuncArgs)
             
            choicesTestingClass = [readoutNeuronsClasses[choice] for choice in choices]
                
            perfsAllFold.append(perfs)
            desOutClassAllFold.append(desOutputClass)
            choicesAllFold.append(choicesTestingClass)
            if returnActivity:
                testInputAllFold.append(validnInputs)
                readoutAllFold.append(readoutActivity)
                reservoirAllFold.append(resActivity)
                desOutputAllFold.append(validnDesOut)
            if trainingPerfs:
                perfsTrainingAllFold.append(perfsTraining)
                desOutClassTrainingAllFold.append(trainDesOutputClass)
                choicesTrainingAllFold.append(choicesTrainingClass)
                if returnActivity:
                    testInputTrainingAllFold.append(trainInputs)
                    readoutTrainingAllFold.append(readoutActivityTraining)
                    reservoirTrainingAllFold.append(resActivityTraining)
                    desOutputTrainingAllFold.append(trainDesOut)
        
        testingResults = {'perfs': perfsAllFold,
                          'desired class': desOutClassAllFold,
                          'choices': choicesAllFold}
        if returnActivity:
            testingResults.update({'inputs': testInputAllFold,
                                   'reservoir activity': reservoirAllFold,
                                   'readout activity': readoutAllFold,
                                   'desired output': desOutputAllFold})
        if trainingPerfs:
            trainingResults = {'perfs': perfsTrainingAllFold,
                               'desired class': desOutClassTrainingAllFold,
                               'choices': choicesTrainingAllFold}
            if returnActivity:
                trainingResults.update({'inputs': testInputTrainingAllFold,
                                        'reservoir activity': reservoirTrainingAllFold,
                                        'readout activity': readoutTrainingAllFold,
                                        'desired output': desOutputTrainingAllFold})
        
        if trainingPerfs:
            return trainingResults, testingResults
        else:
            return testingResults
    
    def cross_validation_with_recomb(self,
                                     inputByClass,
                                     desiredOutByClass,
                                     performanceFunc,
                                     validationSize=0.2,
                                     perfFuncArgs=None,
                                     trainParams=None,
                                     trainingPerfs=False,
                                     seed=None,
                                     returnActivity=False):
        '''Just what it looks to be!
        
        Arguments:
        - inputByClass: a dictionary whose keys are the class names and the
            values are lists containing the different trial inputs in matrices:
            {'className'} [nbTrialsPerClass] <np.ndarray nbTimeSteps * nbInputs>
        - performanceFunc: the function that compute the performances of each
            output TO BE COMPLETED
        - validationSize: a float between 0 and 1 for the proportion of the set
            that is used for cross-validation
        - perfFuncArgs: additional arguments passed to the performanceFunc
        - trainParams: additional arguments passed to the train method
        - seed: NOT USED YET
        - returnActivity: if set to True, return all the data relative to each
            test in each fold in lists: test inputs, reservoir activity in test,
            readout activity in test, desired output in test.
        
        Returns:
        - perfsAllFold: performance for each fold test in the cross-validation
        Additional arguments if the returnActivity switch is activated:
        - testInputAllFold: the inputs of each fold to test the reservoir
        - reservoirAllFold: reservoir activity of each fold to test the reservoir
        - readoutAllFold: readout activity of each fold to test the reservoir
        - desOutputAllFold: desired output of each fold to test the reservoir
        '''
        nbTrials = len(inputByClass.values()[0])
        
        if True in [len(trials) != nbTrials for trials in inputByClass.values()]:
            msg = 'There must be the same number of trials for each class'
            raise ValueError, msg
        nbValidTrials = int(np.ceil(validationSize * nbTrials))
        nbFold = int(np.ceil(nbTrials/nbValidTrials))
        perfsAllFold = []
        desOutClassAllFold = []
        choicesAllFold = []
        
        if returnActivity:
            testInputAllFold = []
            readoutAllFold = []
            reservoirAllFold = []
            desOutputAllFold = []
        if trainingPerfs:
            perfsTrainingAllFold = []
            desOutClassTrainingAllFold = []
            choicesTrainingAllFold = []
            if returnActivity:
                testInputTrainingAllFold = []
                readoutTrainingAllFold = []
                reservoirTrainingAllFold = []
                desOutputTrainingAllFold = []
        
        for fold in range(nbFold):
            trainInputs = []
            trainInputsTmp = []
            validnInputs = []
            trainDesOut = []
            validnDesOut = []
            desOutputClass = []
            trainDesOutputClass = []

            for class_ in inputByClass:
                trainInputsTmp = inputByClass[class_][:fold*nbValidTrials]
                trainInputsTmp += inputByClass[class_][(fold+1)*nbValidTrials:]
                nbTrainingTrials = len(trainInputsTmp)
                trainInputs += trainInputsTmp
                trainDesOut += [desiredOutByClass[class_]] * nbTrainingTrials
                trainDesOutputClass += [class_] * nbTrainingTrials
                validnInputs += inputByClass[class_][fold*nbValidTrials:(fold+1)*nbValidTrials]
                validnDesOut += [desiredOutByClass[class_]] * nbValidTrials
                desOutputClass += [class_] * nbValidTrials
            
            
            
            print 'fold : ' + str(fold)
            self.train(trainInputs,
                       trainDesOut,
                       **trainParams)
            
            if trainingPerfs:
                (perfsTraining,
                 choicesTraining,
                 readoutActivityTraining,
                 resActivityTraining) = self.test_classification(trainInputs,
                                                                 trainDesOut,
                                                                 performanceFunc,
                                                                 perfFuncArgs=perfFuncArgs)
            
            (perfs,
             choices,
             readoutActivity,
             resActivity) = self.test_classification(validnInputs,
                                                     validnDesOut,
                                                     performanceFunc,
                                                     perfFuncArgs=perfFuncArgs)
                
            perfsAllFold.append(perfs)
            desOutClassAllFold.append(desOutputClass)
            choicesAllFold.append(choices)
            if returnActivity:
                testInputAllFold.append(validnInputs)
                readoutAllFold.append(readoutActivity)
                reservoirAllFold.append(resActivity)
                desOutputAllFold.append(validnDesOut)
            if trainingPerfs:
                perfsTrainingAllFold.append(perfsTraining)
                desOutClassTrainingAllFold.append(trainDesOutputClass)
                choicesTrainingAllFold.append(choicesTraining)
                if returnActivity:
                    testInputTrainingAllFold.append(trainInputs)
                    readoutTrainingAllFold.append(readoutActivityTraining)
                    reservoirTrainingAllFold.append(resActivityTraining)
                    desOutputTrainingAllFold.append(trainDesOut)
        
        testingResults = {'perfs': perfsAllFold,
                          'desired class': desOutClassAllFold,
                          'choices': choicesAllFold}
        if returnActivity:
            testingResults.update({'inputs': testInputAllFold,
                                   'reservoir activity': reservoirAllFold,
                                   'readout activity': readoutAllFold,
                                   'desired output': desOutputAllFold})
        if trainingPerfs:
            trainingResults = {'perfs': perfsTrainingAllFold,
                               'desired class': desOutClassTrainingAllFold,
                               'choices': choicesTrainingAllFold}
            if returnActivity:
                trainingResults.update({'inputs': testInputTrainingAllFold,
                                        'reservoir activity': reservoirTrainingAllFold,
                                        'readout activity': readoutTrainingAllFold,
                                        'desired output': desOutputTrainingAllFold})
        
        if trainingPerfs:
            return trainingResults, testingResults
        else:
            return testingResults
    
    def leave_one_out(self,
                      inputList,
                      desiredOutputList,
                      performanceFunc,
                      perfFuncArgs=None,
                      trainParams=None,
                      returnActivity=False):
        '''Just what it looks to be!
        
        Arguments:
        - inputList: the list of all inputs to be used in the leave one out
        - 
        '''
        nbInputs = len(inputList)
        
        perfsAllFold = []
        if returnActivity:
            readoutAllFold = []
            reservoirAllFold = []
        
        for fold in range(nbInputs):
            tmpRemovedInput = inputList.pop(0)
            tmpRemovedDesOut = desiredOutputList.pop(0)
            
            self.train(inputList,
                       desiredOutputList,
                       **trainParams)
            
            perfFuncArgs['fold'] = fold
            [perfs,
             readoutActivity,
             resActivity] = self.test([tmpRemovedInput],
                                      [tmpRemovedDesOut],
                                      performanceFunc,
                                      perfFuncArgs=perfFuncArgs)
            
            perfsAllFold.append(perfs)
            if returnActivity:
                readoutAllFold.append(readoutActivity)
                reservoirAllFold.append(resActivity)
            
            inputList.append(tmpRemovedInput)
            desiredOutputList.append(tmpRemovedDesOut)
            
        if returnActivity:
            return perfsAllFold, readoutAllFold, reservoirAllFold
        else:
            return perfsAllFold

