from Env import Env
from MDP import MDP
import numpy as np
import math

class TestEnv():

    def __init__(self, gamma, environment, mdp):
        self.gamma = gamma
        self.environment = environment
        self.mdp = mdp
        self.evaluateMC(self.gamma, self.environment, self.mdp.getOptimalPolicy(self.gamma))

    # Evaluate policy using monte carlo returns
    def evaluateMC(self, gamma, environment, policy):
        samples = 30000
        R = []
        for s in range(samples):
            r = 0
            curState = environment.reset()  # initalize current state
            episodeEnd = False

            while (episodeEnd != True):

                curStateLocation = environment.getLocation(curState)
                curStateTime = environment.getTime(curState)
                curStateInspectionBits = environment.getInspectionBits(curState)
                curStateOld = environment.getOld(curState)

                find = self.mdp.hasStateWith(curStateLocation, curStateTime, curStateInspectionBits, curStateOld)
                assert find != False

                probs = policy[find]
                sampleAction = np.random.multinomial(1, probs)
                act = np.argmax(sampleAction)

                (nextState, rew, episodeEnd) = environment.step(act)
                r += rew

                curState = nextState
            R.append(r)
        print 'avg reward:', np.mean(R)