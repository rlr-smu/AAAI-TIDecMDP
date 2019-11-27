class Event:
    def __init__(self, agent, pevents, index, name, site, startTime, startTimeEnd):
        self.agent = agent
        self.pevents = pevents
        self.index = index
        self.name = name
        self.site = site
        self.startTime = startTime
        self.startTimeEnd = startTimeEnd

    def __repr__(self):
        return "E: ( " + str(self.agent) + " " + str(self.pevents) + " " + str(self.name) + " " + str(self.site) +  " " + str(self.startTime) + " " + str(self.startTimeEnd) + " )"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.agent == other.agent and self.index==other.index and self.name==other.name and self.site==other.site and self.startTime==other.startTime and self.startTimeEnd==other.startTimeEnd
        return False