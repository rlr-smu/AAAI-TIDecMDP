class Action:
    def __init__(self, ind, name):
        self.index = ind
        self.name = name

    def __repr__(self):
        return "Index: " + str(self.index) + " Name: " + str(self.name)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.index == other.index and self.name==other.name
        return False