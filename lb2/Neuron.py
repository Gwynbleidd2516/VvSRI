import numpy as np

def func(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -500, 500)))

def derivativeFunc(value):
    return np.exp(-value)/np.pow(1-np.exp(-value), 2)
    

class Neuron:
    def __init__(self):
        self.val=0
    
    
    def setValue(self, value):
        try:
            self.val=func(value)
        except RuntimeWarning as _:
            pass