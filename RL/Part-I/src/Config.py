import math
import random
import numpy
import pickle
from Line import Line

class Config:
    def __init__(self, experiment, args=None):
        self.flag = None

        self.experiment = experiment
        self.offset = 500
        self.timeout = 200
        self.workDir = "/home/tarun/RL-PYNB/"
        self.theta = 0.1
        self.gamma = 0.9999
        self.initialxval = 0.00001
        self.alpha = 0.7
        self.beta = 0.7
        self.numIterations = 15000
        self.loggingThreshold = 100
        self.savingThreshold = 1000
        self.runWithSavedModel = False
        self.BaslinelearningRate = 0.0001
        self.PolicylearningRate = 0.0001
        self.batch = 30
        self.numUnitsPerLayer = 32
        self.randomActionProb = 0.0
        self.drate = 0.0
        self.VFLayerNorm = False
        self.PolLayerNorm = False

        self.deltaFinal = 1e-5

        self.agents = None
        self.numberOfLines = None
        self.nPrivatePerLine = None
        self.nShared = None
        self.nLocs = None
        self.minLinesPerSharedLocation = None
        self.maxLinesPerSharedLocation = None
        self.minT = None
        self.maxT = None
        self.minTaction = None
        self.maxTaction = None
        self.maxLocationsPerLinese = None
        self.repeatKTimeSteps = None
        self.lines = None

        self.auction = None
        self.sharedSites = None
        self.agentWiseLines = None

        self.actionTimes = None
        self.T = None
        self.rewardCollection = None
        self.creward = None

        self.rmi = 5
        self.rma = 20

        self.Rmax = None
        self.Rmin = None

        if args != None:
            agents, numberOfLines, nPrivatePerLine, nShared, minLinesPerSharedLocation, maxLinesPerSharedLocation, minT, maxT, minTaction, maxTaction, repeatKTimeSteps = args
            self.flag = 0
            self.agents = agents
            self.numberOfLines = numberOfLines
            self.nPrivatePerLine = nPrivatePerLine
            self.nShared = nShared
            self.minLinesPerSharedLocation = minLinesPerSharedLocation
            self.maxLinesPerSharedLocation = maxLinesPerSharedLocation
            self.minT = minT
            self.maxT = maxT
            self.minTaction = minTaction
            self.maxTaction = maxTaction
            self.lines = []
            self.repeatKTimeSteps = repeatKTimeSteps
            self.generateDomain()
        else:
            self.flag = 1
            self.readConfig()

    def generateDomain(self):
        self.nLocs = (self.numberOfLines * self.nPrivatePerLine) + self.nShared
        self.maxLocationsPerLine = int(math.ceil(float(self.nShared * self.maxLinesPerSharedLocation) / float(self.numberOfLines)))
        self.maxLocationsPerLine += self.nPrivatePerLine;

        self.auction = [-1] * self.nLocs
        self.sharedSites = []
        self.agentWiseLines = []

        print "Experiment: ", self.experiment
        print "Theta: ", self.theta
        print "gamma: ", self.gamma
        print "initialx: ", self.initialxval
        print "alpha: ", self.alpha
        print "beta: ", self.beta
        print "deltaFinal: ", self.deltaFinal
        print "\nagents: ", self.agents
        print "Number of Lines: ", self.numberOfLines
        print "PrivatePerLine: ", self.nPrivatePerLine
        print "nShared: ", self.nShared
        print "nLocs: ", self.nLocs
        print "Agent Max: ", self.maxLocationsPerLine

        for i in xrange(0, self.numberOfLines):
            lst = Line(i)
            for j in xrange(0, self.nPrivatePerLine):
                num = random.randint(0, self.nLocs - 1)
                while self.auction[num] != -1:
                    num = random.randint(0, self.nLocs - 1)
                self.auction[num] = i
                lst.locations.append(num)
            self.lines.append(lst)


        stores = numpy.array([self.maxLocationsPerLine]*self.numberOfLines)

        for i in xrange(0, self.nLocs):
            if self.auction[i] != -1:
                print "Location " + str(i) + " Auctioned to: " + str(self.auction[i])
                continue

            tobesharedbetween = random.randint(self.minLinesPerSharedLocation, self.maxLinesPerSharedLocation)
            assert tobesharedbetween >= self.minLinesPerSharedLocation
            assert tobesharedbetween <= self.maxLinesPerSharedLocation

            setOfLines = set()
            sort_index = numpy.argsort(-stores)
            for j in xrange(0, tobesharedbetween):
                line = sort_index[j]
                setOfLines.add(line)
                stores[line] -= 1

            for line in setOfLines:
                self.lines[line].locations.append(i)
                self.lines[line].sharedLocations.append(i)

            assert len(setOfLines) == tobesharedbetween
            self.auction[i] = setOfLines
            print "Location " + str(i) + " Auctioned to: " + str(setOfLines)
            self.sharedSites.append(i)

        print "Auctioned: ", self.auction
        print "SharedSites: ", self.sharedSites

        line_to_give = 0
        for i in xrange(0, self.agents):
            num = line_to_give
            self.lines[num].owners.append(i)
            self.agentWiseLines.append(self.lines[num])
            line_to_give += 1
            if line_to_give % self.numberOfLines == 0:
                line_to_give = 0

        print "\n=====LINES====="
        for line in self.lines:
            print line

        print "\n==== AGENT WISE LINE ====="
        for i in xrange(0, self.agents):
            print "Agent "+str(i)+" : Line : "+ str(self.agentWiseLines[i].index)

        self.actionTimes = []
        self.rewardCollection = []
        self.creward = []

        totalPow = random.randint(self.minT, self.maxT)
        self.T = 2 ** totalPow

        for i in xrange(0, self.agents):
            t = 2 ** random.randint(self.minTaction, self.maxTaction)
            self.actionTimes.append(t)

        for i in xrange(0, self.agents):
            lst = []
            nlocs = self.agentWiseLines[i].getNumberOfLocations()
            locsact = self.agentWiseLines[i].locations
            for j in xrange(0, nlocs):
                rew = random.randint(self.rmi, self.rma)
                lst.append(rew)
            self.rewardCollection.append(lst)

        for x in self.sharedSites:
            rew = random.randint(2.0 * self.rma, 3.0 * self.rma)
            self.creward.append(rew)

        #Scaling rewards
        

        print "\n\nTotalTime: ", self.T
        print "Action: ", self.actionTimes
        print "MDPRew: ", self.rewardCollection
        print "ConsReward: ", self.creward

        self.R_min = min(self.creward)
        self.R_max = max(self.creward)
        for i in xrange(0, self.agents):
            mmin = min(self.rewardCollection[i])
            mmax = max(self.rewardCollection[i])
            if mmin < self.R_min:
                self.R_min = mmin
            if mmax > self.R_max:
                self.R_max = mmax

        print "Rmin: ", self.R_min
        print "Rmax: ", self.R_max
        self.writeConfig()

    def writeConfig(self):
        with open(self.workDir + 'Data/config' + str(self.experiment) + '.pickle', 'w') as f:
            pickle.dump([self.agents, self.nPrivatePerLine, self.nShared, self.nLocs, self.auction, self.lines,
                         self.agentWiseLines, self.sharedSites, self.T, self.actionTimes,
                         self.rewardCollection, self.creward, self.R_min, self.R_max, self.repeatKTimeSteps], f)
        f.close()

    def readConfig(self):
        with open(self.workDir + 'Data/config' + str(self.experiment) + '.pickle') as f:  # Python 3: open(..., 'rb')
            self.agents, self.nPrivatePerLine, self.nShared, self.nLocs, self.auction, self.lines, \
            self.agentWiseLines, self.sharedSites, self.T, self.actionTimes, self.rewardCollection, self.creward, self.R_min, self.R_max, \
            self.repeatKTimeSteps = pickle.load(f)
        f.close()

        print "Experiment: ", self.experiment
        print "Theta: ", self.theta
        print "gamma: ", self.gamma
        print "initialx: ", self.initialxval
        print "alpha: ", self.alpha
        print "beta: ", self.beta
        print "deltaFinal: ", self.deltaFinal
        print "\nagents: ", self.agents
        print "Number of Lines: ", self.numberOfLines
        print "PrivatePerLine: ", self.nPrivatePerLine
        print "nShared: ", self.nShared
        print "nLocs: ", self.nLocs
        print "Auctioned: ", self.auction
        print "SharedSites: ", self.sharedSites

        print "\n=====LINES====="
        for line in self.lines:
            print line

        print "\n==== AGENT WISE LINE ====="
        for i in xrange(0, self.agents):
            print "Agent "+str(i)+" : Line : "+ str(self.agentWiseLines[i].index)

        print "\n\nTotalTime: ", self.T
        print "Action: ", self.actionTimes
        print "MDPRew: ", self.rewardCollection
        print "ConsReward: ", self.creward
        print "Rmin: ", self.R_min
        print "Rmax: ", self.R_max
        print "RepeatKTimeSteps: ", self.repeatKTimeSteps
