class Line:

    def __init__(self, index):
        self.index = index
        self.locations = []
        self.owners = []
        self.sharedLocations = []

    def getLocations(self):
        return self.locations

    def getOwners(self):
        return self.owners

    def getSharedLocations(self):
        return self.sharedLocations

    def getNextLocation(self, currentLocation):

        currentIndex = self.hasLocation(currentLocation)
        if (currentIndex == False):
            return None

        if (currentIndex + 1 < len(self.locations)):
            return self.locations[currentIndex + 1]
        else:
            return currentLocation

    def getPreviousLocation(self, currentLocation):

        currentIndex = self.hasLocation(currentLocation)
        if (currentIndex == False):
            return None

        if (currentIndex - 1 >= 0):
            return self.locations[currentIndex - 1]
        else:
            return currentLocation

    def getIndexOfLocation(self, location):
        try:
            currentIndex = self.locations.index(location)
            return currentIndex
        except ValueError:
            return -1

    def hasLocation(self, location):
        return location in self.locations

    def isOwner(self, owner):
        return owner in self.owners

    def isShared(self, location):
        return self.hasLocation(location) and location in self.sharedLocations

    def getNumberOfLocations(self):
        return len(self.locations)

    def __repr__(self):
        return "Index: " + str(self.index) + " Owners: " + str(self.owners) + " Locations: " + str(self.locations) + " Shared: " + str(self.sharedLocations)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.index == other.index and self.locations==other.locations and self.owners==other.locations and self.sharedLocations==other.sharedLocations
        return False

