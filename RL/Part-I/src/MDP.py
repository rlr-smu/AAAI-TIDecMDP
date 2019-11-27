from State import State
from Action import Action
import itertools
import random
import numpy as np
import math
import time
import pickle

class MDP:

    def __init__(self, agent, config):
        self.T = config.T
        self.line = config.agentWiseLines[agent]
        self.locs = self.line.getLocations()
        self.nlocs = self.line.getNumberOfLocations()
        self.agent = agent
        self.lst = list(itertools.product([0, 1], repeat=self.nlocs))
        self.states = []
        self.actions = []
        self.actionTime = config.actionTimes[agent]
        self.alpha = config.alpha
        self.beta = config.beta
        self.repeatEveryKTimeSteps = config.repeatKTimeSteps
        self.start = []
        self.policy = None
        self.terminal = None
        self.numberStates = None
        self.numberActions = None
        self.numberVariables = None
        self.config = config

    def generateDomain(self):
        self.terminal = State(0, None, None, None, None, None)
        self.states.append(self.terminal)
        self.initiateActions()
        self.initiateStates()
        self.waste()
        self.reindexStates()
        self.checkTransitionProbabilitySumTo1()
        self.writeTransitions()
        self.writeRewards()
        self.serializeDomain()
        self.defineStart()
        self.numberStates = len(self.states)
        self.numberActions = len(self.actions)
        self.numberVariables = self.numberStates * self.numberActions

    def initiateActions(self):
        self.actions.append(Action(0, "Inspect"))
        self.actions.append(Action(1, "Next"))
        self.actions.append(Action(2, "Previous"))

    def initiateStates(self):
        index = 1
        for i in xrange(0, self.T + 1, self.actionTime):
            for j in xrange(0, self.nlocs):
                for k in self.lst:
                    if k[j] == 0:
                        lyst = [0]
                    elif k[j] == 1:
                        lyst = [0, 1]
                    for t in lyst:
                        st = State(index, j, i, k, t, self.actions)
                        self.states.append(st)
                        index = index + 1

    def transition(self, s, a, sd):

        if s == self.terminal:
            if sd == s:
                return 1
            else:
                return 0

        if s.time == 0:
            if sd == self.terminal:
                return 1
            else:
                return 0

        if sd == self.terminal:
            return 0.0

        if a.name == "Inspect":
            return self.transitionInspect(s, sd)
        else:
            return self.transitionMove(s, sd, a)

    def waste(self):
        iter = 1
        removed = self.removeWasteStates(iter)
        while len(removed) != 0:
            iter += 1
            removed = self.succRemoval(removed, iter)

    def removeWasteStates(self, iter):
        wastestates = []
        sum = 0.0
        val = 0.0
        start = time.time()
        tots = len(self.states)
        offset = 500
        for i in self.states:
            sum += 1
            if sum % offset == 0:
                end = time.time()
                val += float(end - start)
                ntimes = float(sum) / float(offset)
                avg = float(val) / float(ntimes)
                timerem = (float(tots - sum) / float(offset)) * avg
                print "Done. " + str(sum) + " Out of: " + str(tots) + " Avg: " + str(avg) + " Rem: " + str(timerem)
                start = time.time()

            if i == self.terminal:
                continue

            if i.dold == 1 and i.dvals[i.location] == 0:
                wastestates.append(i)
                continue

            sameds = all([i.dvals[j] == 0 for j in xrange(0, self.nlocs)])
            if i.time == self.T and sameds == True and i.dold == 0:
                continue

            flag = 0
            for j in self.states:
                for k in self.actions:
                    if self.transition(j, k, i) != 0:
                        flag = 1
                        break
                if flag == 1:
                    break
            if flag == 0:
                wastestates.append(i)

        for x in wastestates:
            self.states.remove(x)

        print "Iter " + str(iter) + " done and removed " + str(len(wastestates)) + "."
        return wastestates

    def succRemoval(self, removedSt, iter):

        wasteStates = []
        maybe = []

        sumd = 0
        val = 0.0
        start = time.time()
        tots = len(removedSt)
        offset = 500
        for rms in removedSt:

            sumd += 1
            if sumd % offset == 0:
                end = time.time()
                val += float(end - start)
                ntimes = float(sumd) / float(offset)
                avg = float(val) / float(ntimes)
                timerem = (float(tots - sumd) / float(offset)) * avg
                print "Done. " + str(sumd) + " Out of: " + str(tots) + " Avg: " + str(avg) + " Rem: " + str(timerem)
                start = time.time()

            for sts in self.states:

                if sts == self.terminal:
                    continue

                if sts.dold == 1 and sts.dvals[sts.location] == 0:
                    wasteStates.append(sts)
                    continue

                sameds = all([sts.dvals[j] == 0 for j in xrange(0, self.nlocs)])
                if sts.time == self.T and sameds == True and sts.dold == 0:
                    continue

                for a in self.actions:
                    if self.transition(rms, a, sts) != 0:
                        maybe.append(sts)
                        break

        for i in maybe:
            flag = 0
            for j in self.states:
                for k in self.actions:
                    if self.transition(j, k, i) != 0:
                        flag = 1
                        break
                if flag == 1:
                    break
            if flag == 0:
                wasteStates.append(i)

        prevs = []
        for torem in wasteStates:
            if torem in self.states:
                prevs.append(torem)
                self.states.remove(torem)

        print "Iter " + str(iter) + " done and removed " + str(len(prevs)) + "."
        return prevs

    def checkTransitionProbabilitySumTo1(self):
        for k in self.actions:
            for i in self.states:
                sum = 0
                for j in self.states:
                    tran = self.transition(i, k, j)
                    sum += tran
                if (sum != 1):
                    print "WARNING: k: " + str(k) + " i: " + str(i) + " Sum: " + str(sum)

    def transitionInspect(self, s, sd):

        if s.location != sd.location:
            return 0

        if sd.time != s.time - self.actionTime:
            return 0

        if sd.time < 0:
            return 0

        sameds = all([s.dvals[j] == sd.dvals[j] for j in xrange(0, self.nlocs) if j != s.location])

        if (self.isReset(s)):
            sameds = all(sd.dvals[j] == 0 for j in xrange(0, self.nlocs) if j != sd.location)

        if sameds == False:
            return 0

        if sd.dold != s.dvals[s.location]:
            return 0

        if s.dold == 1 and s.dvals[s.location] == 0:
            return 0

        if s.dvals[s.location] == 0:
            return self.alpha if sd.dvals[sd.location] == 1 else (1 - self.alpha)
        elif s.dvals[s.location] == 1:
            return 1 if sd.dvals[sd.location] == 1 else 0
        else:
            return 0

    # defines the next location agent can travel to.
    def getNextLocation(self, location):
        if (location + 1) < self.nlocs:
            return location + 1
        else:
            return location

    # defines previous location agent can travel to.
    def getPreviousLocation(self, location):
        if (location - 1) >= 0:
            return location - 1
        else:
            return location

    def transitionMove(self, s, sd, action):

        l = s.location
        ld = sd.location
        t = s.time
        td = sd.time

        dold = s.dold
        doldd = sd.dold

        dest = None
        if (action.name == "Next"):
            dest = self.getNextLocation(s.location)
        elif (action.name == "Previous"):
            dest = self.getPreviousLocation(s.location)

        if (dest is None):
            return 0.0

        sameds = all([s.dvals[j] == sd.dvals[j] for j in xrange(0, self.nlocs)])

        if (self.isReset(s)):
            sameds = all(sd.dvals[j] == 0 for j in xrange(0, self.nlocs))

        if sameds == False:
            return 0

        if (ld != dest and ld != l):
            return 0

        if td != t - self.actionTime:
            return 0

        if td < 0:
            return 0

        if doldd != sd.dvals[ld]:
            return 0

        if dold == 1 and s.dvals[l] == 0:
            return 0

        if ld == l and ld == dest:
            return 1.0
        elif ld == l:
            return 1 - self.beta
        elif ld == dest:
            return self.beta
        else:
            return 0.0

    def rewardFunction(self, s, a):
        if s == self.terminal:
            return 0.0
        if s.dold == 0 and s.dvals[s.location] == 1:
            return self.config.rewardCollection[self.agent][s.location]
        else:
            return 0.0

    def call(self):
        import datetime
        print "Start: ",datetime.datetime.now()
        if self.config.flag == 0:
            self.generateDomain()
        else:
            self.deserializeDomain()
            self.defineStart()
            self.numberStates = len(self.states)
            self.numberActions = len(self.actions)
            self.numberVariables = self.numberStates * self.numberActions
        print "End: ",datetime.datetime.now()
        return self

    def reindexStates(self):
        index = 0
        lst = []
        for x in self.states:
            a = State(index, x.location, x.time, x.dvals, x.dold, self.actions)
            a.setReward(x.reward)
            a.setTransition(x.transition)
            lst.append(a)
            index = index + 1
        self.states = lst

    def serializeDomain(self):
        f = open(self.config.workDir + "Data/Domain" + str(self.agent) + "_exp_" + str(self.config.experiment) + ".pickle", 'w')
        pickle.dump(self.states, f)
        pickle.dump(self.actions, f)
        for s in self.states:
            pickle.dump(s.transition, f)
            pickle.dump(s.reward, f)
        f.close()

    def deserializeDomain(self):
        f = open(self.config.workDir + "Data/Domain" + str(self.agent) + "_exp_" + str(self.config.experiment) + ".pickle")
        self.states = pickle.load(f)
        self.terminal = self.states[0]
        self.actions = pickle.load(f)
        for s in self.states:
            s.transition = pickle.load(f)
            s.reward = pickle.load(f)
        f.close()

    def defineStart(self):
        sum = 0
        for i in self.states:
            if i==self.terminal:
                continue
            sameds = all([i.dvals[j] == 0 for j in xrange(0, self.nlocs)])
            if i.time == self.T and sameds == True and i.dold == 0:
                sum += 1
        for i in self.states:
            if i==self.terminal:
                self.start.append(float(0))
                continue
            sameds = all([i.dvals[j] == 0 for j in xrange(0, self.nlocs)])
            if i.time == self.T and sameds == True and i.dold == 0:
                self.start.append(float(1 / float(sum)))
            else:
                self.start.append(float(0))

    def isReset(self, state):
        curTime = state.time
        remTime = curTime - self.actionTime
        if (remTime >= 0):
            n = (float(self.T) - float(remTime)) / float(self.repeatEveryKTimeSteps)
            int_n = int(n)
            if (int_n == n):
                return True
        return False

    def writeTransitions(self):
        print "     Writing Transitions for Agent " + str(self.agent)
        for a in self.actions:
            for s in self.states:
                for sd in self.states:
                    tt = self.transition(s, a, sd)
                    if tt != 0:
                        s.transition.append((a.index, sd.index, tt))

    def writeRewards(self):
        print "     Writing Rewards for Agent " + str(self.agent)
        for s in self.states:
            for a in self.actions:
                reward = self.rewardFunction(s, a)
                s.reward.append((a.index, reward))

    def generateLPAc(self, gamma, genA=False):
        print "Generating LP "
        A_mat = []
        if genA is True:

            decisionvar = []
            for x in self.states:
                triple = []
                for y in self.states:
                    triplet = []
                    for a in y.possibleActions:
                        if x.index == y.index:
                            triplet.append(float(1))
                        else:
                            triplet.append(float(0))
                    triple.append(triplet)
                decisionvar.append(triple)

            for x in self.states:
                incoming = []
                for s in self.states:
                    for t in s.transition:
                        if t[1] == x.index and t[2] != 0:
                            incoming.append((s, t[0], t[2]))

                for h in incoming:
                    decisionvar[x.index][h[0].index][h[1]] -= gamma * float(h[2])

            A_mat = []
            for x in decisionvar:
                lit = []
                for t in x:
                    lit.extend(t)
                A_mat.append(lit)

        R_mat = []
        for x in self.states:
            for y in x.possibleActions:
                for r in x.reward:
                    if r[0] == y.index:
                        R_mat.append(r[1])
        return np.array(A_mat), np.array(R_mat)

    def solveLP(self, gamma):
        A, R = self.generateLPAc(gamma, genA=True)
        print np.shape(A)
        print np.shape(R)

        R_mat = np.array(R)[np.newaxis].T
        A_mat = np.array(A)

        alpha = self.start

        x = cvxpy.Variable(self.numberVariables, 1)
        obj = cvxpy.Maximize(np.transpose(R_mat)*x)
        constraints = [A_mat*x == alpha, x >= 0.00001]
        prob = cvxpy.Problem(obj, constraints)
        prob.solve()
        #print "status:", prob.status
        print "LPsolver: optimal value", prob.value
        #print "Optimal x: ", x.value
        print "Sum of x values: ", cvxpy.sum_entries(x).value
        return x.value

    def getOptimalPolicy(self, gamma):
        self.policy = dict()
        xvals = self.solveLP(gamma)
        for s in self.states:
            xs = [xvals[(s.index * self.numberActions) + a.index] for a in self.actions]
            xs_sum = sum(xs)
            probs = [float(val)/float(xs_sum) for val in xs]
            assert sum(probs) - 1 <= 1e-5
            self.policy[s.index] = probs
        return self.policy

    def hasStateWith(self, location, time, dvals, dold):
        for s in self.states:
            if s.location == location and s.time == time and all(s.dvals[vals] == dvals[vals] for vals in xrange(0, len(dvals))) and s.dold == dold:
                assert s != self.terminal
                return s.index
        return False

    def hasStateWithActualLocation(self, actlocation, time, dvals, dold):
        ind = self.line.getIndexOfLocation(actlocation)
        assert ind != -1
        for s in self.states:
            if s.location == ind and s.time == time and all(s.dvals[vals] == dvals[vals] for vals in xrange(0, len(dvals))) and s.dold == dold:
                assert s != self.terminal
                return s.index
        return False
