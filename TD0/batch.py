# A class for the replay memory
from collections import deque
import numpy as np


class MemoryBuffer:
    "An experience replay buffer using numpy arrays"
    "state_shape = (2,) for mountain car"
    def __init__(self, length, state_shape):
        self.length = length
        self.state_shape = state_shape
        shape = (length,) + state_shape
        self._s = np.zeros(shape, dtype=np.float32) # starting states
        self._t = np.zeros(length, dtype=np.float32) # actions
        self.index = 0 # points one position past the last inserted element
        self.size = 0 # current size of the buffer
    
    def append(self, s, target):
        self._s[self.index] = s
        self._t[self.index] = target
        self.index = (self.index+1) % self.length
        self.size = np.min([self.size+1,self.length])
    
    
    def minibatch(self, size):
        indices = np.random.choice(self.size, size=size, replace=False)
        x = np.zeros((size,) + self.state_shape)
        y = np.zeros(size)
        for i in range(size):
            x[i] = self._s[indices[i]]
            y[i] = self._t[indices[i]]
        return x, y