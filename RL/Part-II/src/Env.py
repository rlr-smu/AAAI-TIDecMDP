import numpy as np
from copy import deepcopy

class Env:

    def __init__(self, agent, config):
        self.config = config
        self.agent = agent
        self.gamma = self.config.gamma #Discount Factor
        self.epsilon = 1e-5
        self.currentState = [-1]*self.config.batch #Current state of MDP
        self.numberOfLocations = self.config.agentWiseLines[self.agent].getNumberOfLocations() #Number of locations available to agent
        self.T = self.config.T #Time Horizon
        self.validLengthOfState = (2*self.numberOfLocations + self.T + 2) #Valid length of state

        """
        Time taken for any action. Can be made an array for different actions or passed in constructor.
        """
        self.actionTime = self.config.actionTimes[self.agent]

        #Possible start states
        self.startStates = []

        #Probabilities for possible start states
        self.startStateProbs = []

        #Action dictionary
        self.actionDict = dict()
        self.numberOfActions = 3  # 0 - Inspect, 1 - Next, 2 - Previous
        self.actionDict[0] = "Inspect"
        self.actionDict[1] = "Next"
        self.actionDict[2] = "Previous"

        #Defines start states with their probabilities
        self.setStartStates()

        self.alpha = self.config.alpha # Successful Inspection Probability [Can be passed via constructor]
        self.beta = self.config.beta # Successful Movement Probability [Can be passed via constructor]
        self.collectionReward = self.config.rewardCollection[self.agent] # Successful inspection reward at agent's location. Can be different for different locations.

        # Reset transition function every k time steps for re-inspection.
        self.repeatEveryKTimeSteps = self.config.repeatKTimeSteps

        # Resets the environment.
        self.reset()

    def getAction(self, act):
        assert sum(act) == 1.0
        for bit in xrange(0, len(act)):
            if act[bit] == 1.0:
                return bit


    #Is state array valid for given MDP ?
    def isValid(self, state):
        assert len(state) == self.validLengthOfState
        assert self.getLocationBits(state).count(1) == 1
        assert len(self.getLocationBits(state)) == self.numberOfLocations
        assert self.getTimeBits(state).count(1) == 1
        assert len(self.getTimeBits(state)) == self.T + 1
        assert len(self.getInspectionBits(state)) == self.numberOfLocations
        assert (self.getOld(state) == 0 or self.getOld(state) == 1)

    #Get location bits from state array
    def getLocationBits(self, state):
        return state[0:self.numberOfLocations]

    #Get Time Bits from state array
    def getTimeBits(self, state):
        start = self.numberOfLocations
        return state[start:start+self.T+1]

    #Get Inspection Bits from state array
    def getInspectionBits(self, state):
        start = self.numberOfLocations + self.T + 1
        return state[start:start+self.numberOfLocations]

    #Set Inspection Bits of a state and return updated state.
    def setInspectionBits(self, state, inspectBits):
        self.isValid(state)
        start = self.numberOfLocations + self.T + 1
        assert len(inspectBits) == self.numberOfLocations
        for i in inspectBits:
            state[start] = i
            start += 1
        return state

    #Get Old bit from state array
    def getOld(self, state):
        return state[-1]

    # Set Old Bit of a state and return updated state.
    def setOldBit(self, state, oldBit):
        self.isValid(state)
        assert (oldBit == 0 or oldBit == 1)
        state[-1] = oldBit
        return state

    #Get current location from state
    def getLocation(self, state):
        self.isValid(state)
        bits = self.getLocationBits(state)
        return bits.index(1)

    #Get current time from state
    def getTime(self, state):
        self.isValid(state)
        bits = self.getTimeBits(state)
        return bits.index(1)

    #Determine whether two state are equal
    def isEqual(self, state1, state2):
        return (self.getLocationBits(state1) == self.getLocationBits(state2) and
                self.getTimeBits(state1) == self.getTimeBits(state2) and
                self.getInspectionBits(state1) == self.getInspectionBits(state2) and
                self.getOld(state1) == self.getOld(state2))

    #Get state bit representation based on time, location, InspectionBits = 0 and dold = 0
    def getStateRep(self, location, time):
        assert location < self.numberOfLocations and location >= 0
        assert time >= 0 and time <= self.T
        state = [0]*self.validLengthOfState
        state[location] = 1
        state[self.numberOfLocations + time] = 1
        return state

    #define start states and their probabilities
    def setStartStates(self):
        for locs in xrange(0, self.numberOfLocations):
            self.startStates.append(self.getStateRep(locs, self.T))
        self.startStateProbs = [float(1) / float(len(self.startStates))]*len(self.startStates)
        assert float(sum(self.startStateProbs)) - float(1) <= self.epsilon

    def reset(self):
        self.currentState = []
        for i in xrange(self.config.batch):
            sample = np.random.multinomial(1, self.startStateProbs)
            stateIndex = np.argmax(sample)
            currentSt = self.startStates[stateIndex]
            self.currentState.append(currentSt)
        return self.currentState

    # defines the next location agent can travel to.
    def getNextLocation(self, location):
        if (location + 1) < self.numberOfLocations:
            return location + 1
        else:
            return location

    # defines previous location agent can travel to.
    def getPreviousLocation(self, location):
        if (location - 1) >= 0:
            return location - 1
        else:
            return location

    # Define whether a state is a reset state or not depending on repeatEveryKTimeSteps.
    def isReset(self, state):
        curTime = self.getTime(state)
        remTime = curTime - self.actionTime
        if (remTime >= 0):
            n = (float(self.T) - float(remTime)) / float(self.repeatEveryKTimeSteps)
            int_n = int(n)
            if (int_n == n):
                return True
        return False

    # Given current state and action along with rewards, reurn next possible states with probabilities.
    def getNextPossibleStates(self, state, action):

        currentTime = self.getTime(state)
        currentLocation = self.getLocation(state)
        currentInspectionBits = self.getInspectionBits(state)
        currentDold = self.getOld(state)

        if (currentTime == 0 or (currentTime - self.actionTime < 0)):
            return None

        retvals = []

        #Inspect action
        if (action == 0):

            #already inspected Site
            if (currentInspectionBits[currentLocation] == 1):

                if (self.isReset(state)):
                    successState = self.getStateRep(currentLocation, currentTime - self.actionTime)
                    newInspectionBits = [0]*self.numberOfLocations
                    newInspectionBits[currentLocation] = 1
                    successState = self.setInspectionBits(successState, newInspectionBits)
                    successState = self.setOldBit(successState, 1)
                    retvals.append((successState, 1.0, 0.0))
                else:
                    successState = self.getStateRep(currentLocation, currentTime - self.actionTime)
                    successState = self.setInspectionBits(successState, currentInspectionBits)
                    successState = self.setOldBit(successState, 1)
                    retvals.append((successState, 1.0, 0.0))

            #Uninspected Site
            elif (currentInspectionBits[currentLocation] == 0):

                if (self.isReset(state)):
                    successState = self.getStateRep(currentLocation, currentTime - self.actionTime)
                    newInspectionBits = [0]*self.numberOfLocations
                    newInspectionBits[currentLocation] = 1
                    successState = self.setInspectionBits(successState, newInspectionBits)
                    successState = self.setOldBit(successState, 0)
                    retvals.append((successState, self.alpha, float(self.collectionReward[currentLocation])))

                    failureState = self.getStateRep(currentLocation, currentTime - self.actionTime)
                    newInspectionBits = [0]*self.numberOfLocations
                    failureState = self.setInspectionBits(failureState, newInspectionBits)
                    failureState = self.setOldBit(failureState, 0)
                    retvals.append((failureState, 1 - self.alpha, 0.0))

                else:
                    successState = self.getStateRep(currentLocation, currentTime - self.actionTime)
                    newInspectionBits = deepcopy(currentInspectionBits)
                    newInspectionBits[currentLocation] = 1
                    successState = self.setInspectionBits(successState, newInspectionBits)
                    successState = self.setOldBit(successState, 0)
                    retvals.append((successState, self.alpha, self.collectionReward[currentLocation]))

                    failureState = self.getStateRep(currentLocation, currentTime - self.actionTime)
                    failureState = self.setInspectionBits(failureState, currentInspectionBits)
                    failureState = self.setOldBit(failureState, 0)
                    retvals.append((failureState, 1-self.alpha, 0.0))

        # Next action or Previous Action
        elif (action == 1 or action == 2):

            if (action == 1):
                newLocation = self.getNextLocation(currentLocation)
            elif (action == 2):
                newLocation = self.getPreviousLocation(currentLocation)

            #If new location is same as current location.
            if (newLocation == currentLocation):

                if (self.isReset(state)):
                    successState = self.getStateRep(newLocation, currentTime - self.actionTime)
                    newInspectionBits = [0]*self.numberOfLocations
                    successState = self.setInspectionBits(successState, newInspectionBits)
                    successState = self.setOldBit(successState, 0)
                    retvals.append((successState, 1.0, 0.0))
                else:
                    successState = self.getStateRep(newLocation, currentTime - self.actionTime)
                    successState = self.setInspectionBits(successState, currentInspectionBits)
                    successState = self.setOldBit(successState, currentDold)
                    retvals.append((successState, 1.0, 0.0))

            # If new location is not same as current location.
            else:

                if (self.isReset(state)):
                    successState = self.getStateRep(newLocation, currentTime - self.actionTime)
                    newInspectionBits = [0]*self.numberOfLocations
                    successState = self.setInspectionBits(successState, newInspectionBits)
                    successState = self.setOldBit(successState, 0)
                    retvals.append((successState, self.beta, 0.0))

                    failureState = self.getStateRep(currentLocation, currentTime - self.actionTime)
                    newInspectionBits = [0]*self.numberOfLocations
                    failureState = self.setInspectionBits(failureState, newInspectionBits)
                    failureState = self.setOldBit(failureState, 0)
                    retvals.append((failureState, 1 - self.beta, 0.0))

                else:
                    successState = self.getStateRep(newLocation, currentTime - self.actionTime)
                    successState = self.setInspectionBits(successState, currentInspectionBits)
                    successState = self.setOldBit(successState, currentInspectionBits[newLocation])
                    retvals.append((successState, self.beta, 0.0))

                    failureState = self.getStateRep(currentLocation, currentTime - self.actionTime)
                    failureState = self.setInspectionBits(failureState, currentInspectionBits)
                    failureState = self.setOldBit(failureState, currentDold)
                    retvals.append((failureState, 1 - self.beta, 0.0))

        return retvals
    
    def rewardFunction(self, state, action, stated):
        # Inspect action
        possibilities = self.getNextPossibleStates(state, action)
        for rets in possibilities:
            if self.isEqual(rets[0], stated):
                return rets[2]
            else:
                return 0.0

    #Prints a state in human readable format
    def printState(self, state):
        return "Location: " + str(self.getLocation(state)) + " Time : " + str(self.getTime(state)) + " Inspection Status: " + str(self.getInspectionBits(state)) + " Old Bit: " + str(self.getOld(state))

    """
        Run one timestep of the environment's dynamics. When end of episode
        is reached, reset() should be called to reset the environment's internal state.
        
        Input
        -----
        action : an action provided by the environment
        
        Outputs
        -------
        (observation, reward, done, info)
        observation : agent's observation of the current environment
        reward [Float] : amount of reward due to the previous action
        done : a boolean, indicating whether the episode has ended
    """
    def step(self, action):
        rews = []
        assert len(action) == self.config.batch

        for i in xrange(self.config.batch):
            possibleSt = self.getNextPossibleStates(self.currentState[i], action[i])

            if (possibleSt is None):
                # Should be same for all batches.
                return (None, 0.0, True)

            probabilities = [x[1] for x in possibleSt]
            rewards = [x[2] for x in possibleSt]

            sample = np.random.multinomial(1, probabilities)
            newStateIndex = np.argmax(sample)
            newStateTuple = possibleSt[newStateIndex]

            self.currentState[i] = newStateTuple[0]
            rews.append(newStateTuple[2])

        return (self.currentState, rews, False)