import numpy as np

class Grid(object):
    def __init__(self, resolution=1, width=10, height=10, center=[5,5]):
        self.resolution = resolution
        self.width = width
        self.height = height
        self.center = center

        self.x_min = - self.center[0]
        self.x_max = self.width - self.center[0]
        self.y_min = - self.center[1]
        self.y_max = self.height - self.center[1]

        self.grid = np.zeros((int(height/resolution),int(width/resolution)))

        self.UNKNOWN = 0
        self.KNOWN = 1

    def to_list_of_areas(self):
        center_list = []
        for x in range(int(self.width/self.resolution)):
            for y in range(int(self.height/self.resolution)):
                if self.grid[x,y] == self.KNOWN:
                    center_list.append([x-self.center[0], y-self.center[1]])
        return center_list

    def to_grid(self, x, y):
        return [x+self.center[0], y+self.center[1]]
