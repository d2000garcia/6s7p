import numpy as np
from matplotlib import pyplot as plt
pi = np.pi
Mfd = 5
wav = 852
#Mfd^2/wav = order 1/1000
q0 = (pi/4) * Mfd**2/wav /1000
f = [0.002,0.0046,0.0075,0.011,0.0153,0.0184]
np.linspace()