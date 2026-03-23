from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import find_peaks

scan = input("456 or 894?")
if scan == '456':
    data = np.loadtxt(r"C:\Users\wolfw\git\6s7p\SaturationEffects3-17-26\SatEffects3-17-26_456.csv",delimiter=',')
elif scan == '894':
    data = np.loadtxt(r"C:\Users\wolfw\git\6s7p\SaturationEffects3-17-26\SatEffects3-17-26_894.csv",delimiter=',')
xs = data[:,0]
for i in range(21):
    temp = data[:,i+1]
    peaks, properties = find_peaks(-temp,width=500,prominence=0.02)
    plt.plot(xs, temp,'-b')
    plt.vlines(x=xs[peaks], ymin= temp[peaks], ymax = properties['prominences']+temp[peaks], color = "red")
    plt.show()
    plt.clf()