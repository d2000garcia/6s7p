import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from matplotlib import lines as lines
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

def process_beatnote(ogBeat,run_avg_num):
    # dim = data.shape[1]
    # print(data.shape)
    # ogBeat = data[:,dim-2]
    # piezoV = data[:, dim-1]
    # plt.plot(piezoV,ogBeat-min(ogBeat))
    runningavg = np.convolve(ogBeat,np.ones(run_avg_num)/run_avg_num, mode='same')
    corrected=ogBeat-runningavg
    # print(corrected.shape)
    # plt.plot(piezoV,corrected)
    # plt.ylim(-2,5)
    # plt.savefig(r"C:\Users\wolfw\Downloads\BeatnoteProcess7-321-25\456beattest.png")
    # plt.savefig(r"D:\Diego\git\6s7p\894beattest.png")
    # plt.show()
    return corrected
    
def correct_peak(peak_indices, peak_val):
    itterations = 0 
    done = False
    while not done:
        temp = np.mean(peak_val)/2
        length = len(peak_val)
        done = True
        for i in range(length):
            index = length-i-1
            if peak_val[index] < temp:
                if done:
                    itterations += 1
                done = False
                peak_val.pop(index)
                peak_indices.pop(index)
    print('itterations done %i' % (itterations))
    return (peak_indices, peak_val)

    

class data:
    def __init__(self, filename, skip_lines, BeatRunAvgN=100):
        self.pure_dat = simple_dat_get(filename, skip_lines)
        self.Tavg = self.pure_dat[:,0:int(self.pure_dat.shape[1]/2)-1].mean(1)
        self.Pavg = self.pure_dat[:,int(self.pure_dat.shape[1]/2)-1:self.pure_dat.shape[1]-2].mean(1)
        self.beat = self.pure_dat[:,self.pure_dat.shape[1]-2]
        self.indices = self.pure_dat[:,self.pure_dat.shape[1]-1]
        self.indices = self.indices - self.indices.min()
        self.scaledT = self.Tavg/self.Pavg
        peak_indices = find_peaks(self.beat, height=-50, distance=50)
        # peak_val = np.array(list(map(lambda k: self.beat[k], peak_indices[0])))
        peak_val = peak_indices[1]['peak_heights']
        peak_indices = peak_indices[0]
        corrected = process_beatnote(self.beat, BeatRunAvgN)
        standard_peak_min = np.std(corrected[BeatRunAvgN+1:len(self.indices)-BeatRunAvgN-1])*2
        peak_indices2 = find_peaks(corrected, height=standard_peak_min,distance=50)
        peak_indices = peak_indices.tolist()
        peak_val = peak_val.tolist()
        temp = len(peak_indices)
        temp2 = len(self.indices)
        for i in range(len(peak_indices)):
            if peak_indices[temp-i-1] <= BeatRunAvgN or peak_indices[temp-i-1] >= temp2 - BeatRunAvgN:
                peak_indices.pop(temp-i-1)
                peak_val.pop(temp-i-1)
        # peak_val2 = list(map(lambda k: corrected[k], peak_indices2[0]))
        peak_val2 = peak_indices2[1]['peak_heights']
        peak_indices2 = peak_indices2[0]
        (peak_indices2, peak_val2) = correct_peak(peak_indices=peak_indices2.tolist(), peak_val=peak_val2.tolist())
        # print(peak_indices[0])
        # print(peak_indices2[0])
        print(set(peak_indices2).issuperset(set(peak_indices)))
        plt.plot(self.indices,self.scaledT)
        plt.show()
        plt.scatter(self.indices, self.beat)
        plt.scatter(peak_indices,peak_val, color='red', marker='x')
        plt.show()
        plt.scatter(self.indices, corrected)
        plt.scatter(peak_indices2,peak_val2, color='red', marker='x')
        temp = np.std(corrected[BeatRunAvgN+1:len(self.indices)-BeatRunAvgN-1])*2
        temp1 = [0,len(self.indices)]
        temp2 = [temp, temp]
        print(temp1)
        print(temp2)
        plt.plot(temp1,temp2, '-r')
        plt.ylim(0)
        plt.show()
        

data(r"D:\Diego\git\6s7p\894Scan.csv", 0)
        


        
