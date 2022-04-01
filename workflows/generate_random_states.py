import flow
import hoomd
import numpy as np
import signac

class LogTrigger(hoomd.trigger.Trigger):

    def __init__(self):
        hoomd.trigger.Trigger.__init__(self)

    def compute(self, timestep):
        return (timestep**(1 / 2)).is_integer()