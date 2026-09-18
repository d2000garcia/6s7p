import numpy as np
from matplotlib import pyplot as plt
def lin_pol(rad):
    #Angle to x-axis in radians
    return np.array([[np.cos(rad)**2,np.sin(2*rad)/2],[np.sin(2*rad)/2,np.sin(rad)**2]])

def quar_wav(rad):
    #Quarter wave plate with fast axis rad radians from x-axis
    return np.exp(-np.pi*1.j/4) * np.array([[1-(1-1.j)*np.sin(rad)**2,(1-1.j)*np.sin(2*rad)/2],[(1-1.j)*np.sin(2*rad)/2,1.j+(1-1.j)*np.sin(rad)**2]])

def mag(pol):
    return np.sqrt(np.sum(pol*pol.conjugate()))

pol = np.array([1+0.j,0+0.j])
angles = np.linspace(0,2*np.pi,1000)