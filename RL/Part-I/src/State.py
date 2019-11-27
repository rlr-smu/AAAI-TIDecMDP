class State:

    def __init__(self, ind, location, time, dvals, dold, actions ):
        self.index = ind
        self.location = location
        self.time = time
        self.dvals = dvals
        self.dold = dold
        self.possibleActions = actions
        self.transition = []
        self.reward = []

    def setTransition(self, tran):
        self.transition = tran

    def setReward(self, rew):
        self.reward = rew

    def __repr__(self):
        return "Index: " + str(self.index) + " Location: " + str(self.location) + " Time: " + str(self.time) + " Dvals " + str(self.dvals) + " Dold: " + str(self.dold)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.index == other.index and self.location==other.location and self.time==other.time and self.dvals==other.dvals and self.dold==other.dold
        return False