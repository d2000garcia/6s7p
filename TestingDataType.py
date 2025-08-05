import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
def simple_dat_get(filename, skip_lines):
    file = open(filename, 'r')
    data = []
    for i in range(skip_lines):
        file.readline()
    for line in file:
        x = list(map(float, line.strip().split(',')))
        data.append(x)
    # print(data)
    
    return np.array(data)

class data:
    def __init__(self, filename, skip_lines):
        self.pure_dat = simple_dat_get(filename, skip_lines)
        self.Tavg = self.pure_dat[:,0:int(self.pure_dat.shape[1]/2)-1].mean(1)
        self.Pavg = self.pure_dat[:,int(self.pure_dat.shape[1]/2)-1:self.pure_dat.shape[1]-2].mean(1)
        self.beat = self.pure_dat[:,self.pure_dat.shape[1]-2]
        self.indices = self.pure_dat[:,self.pure_dat.shape[1]-1]
        self.indices = self.indices - self.indices.min()
        self.scaledT = self.Tavg/self.Pavg
        peak_indices = find_peaks(self.beat)
        # print(peak_indices[0])
        peak_val = np.array(list(map(lambda k: self.beat[k], peak_indices[0])))
        plt.plot(self.indices,self.scaledT)
        plt.show()
        plt.scatter(self.indices, self.beat)
        plt.scatter(peak_indices[0],peak_val, color='red', marker='x')
        plt.show()
        

data(r"D:\Diego\git\6s7p\456Scan.csv", 0)
        


        
