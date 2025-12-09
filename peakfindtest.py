import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as sig

x = np.linspace(-10,10,10000) 
f = lambda x,x0,sigma: np.exp(-(x-x0)**2/(2*sigma**2))/(sigma*np.log(2*np.pi)) 
h = lambda x: f(x,4,1) + f(x,0,0.5) + f(x,-4,0.25) -5
def thing(x,f,width): 
     y = f(x)
     peaks, properties = sig.find_peaks(y, width=width,prominence=0.5) 
     k1 = x[list(map(lambda k: int(k),properties["left_ips"]))]
     k2= x[list(map(lambda k: int(k),properties["right_ips"]))]
     plt.vlines(x=x[peaks], ymin=min(y), ymax = y[peaks], color = "red")
     plt.hlines(y=properties["width_heights"], xmin=k1,xmax=k2, color = "red")
     print(properties['prominences'])
     print(y[peaks]-min(y))
     plt.plot(x,y)
     plt.show()
thing(x,h,450)  