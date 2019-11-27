class Constraint:
    def __init__(self, Events, rew, index, name):
        self.Events = Events
        self.reward = rew
        self.index = index
        self.name = name

    def __repr__(self):
        return "Cons: ( " + str(self.index) + " " + str(self.Events) + " " + str(self.reward) + " " + str(self.name) + " )"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.index == other.index and self.reward == other.reward and self.name==other.name
        return False