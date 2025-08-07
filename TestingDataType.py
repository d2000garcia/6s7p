import numpy as np
from numpy.polynomial import Polynomial as poly
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from matplotlib import lines as lines
def simple_dat_get(filename, skip_lines=0):
    file = open(filename, 'r')
    data = []
    for i in range(skip_lines):
        file.readline()
    for line in file:
        x = list(map(float, line.strip().split(',')))
        data.append(x)
    # print(data)
    
    return np.array(data)

def process_beatnote(indices,ogBeat,run_avg_num):
    #Initial Peak Finding with base data
    peak_indices = find_peaks(ogBeat, height=-50, distance=50)
    peak_val = peak_indices[1]['peak_heights']
    peak_indices = peak_indices[0]
    #filtering data
    runningavg = np.convolve(ogBeat, np.ones(run_avg_num)/run_avg_num, mode='same')
    filteredBeat=ogBeat-runningavg
    #Peak finding filtered data
    standard_peak_min = np.std(filteredBeat[run_avg_num+1:len(indices)-run_avg_num-1])*2
    peak_indices2 = find_peaks(filteredBeat, height=standard_peak_min,distance=50)
    peak_indices = peak_indices.tolist()
    peak_val = peak_val.tolist()
    #filter out peaks from og data at ends to make comparable
    cutoff_ends(peak_indices, peak_val, run_avg_num, len(indices)-run_avg_num)
    peak_val2 = peak_indices2[1]['peak_heights']
    peak_indices2 = peak_indices2[0]
    (peak_indices2, peak_val2) = correct_peaks(peak_indices=peak_indices2.tolist(), peak_val=peak_val2.tolist())
    return (peak_indices, peak_val, peak_indices2, peak_val2, filteredBeat)

def correct_peaks(peak_indices, peak_val):
    #Taking out edge peaks
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

def cutoff_ends(x, y, mini, maxi):
    #Takes in a list data set y and list indices x and takes off values with indices below min and above max
    temp= len(x)
    for i in range(temp):
        if x[temp-i-1] <= mini or x[temp-i-1] >= maxi:
            x.pop(temp-i-1)
            y.pop(temp-i-1)
    return (x,y)

def LinFit(data_bounds, indices, data):
    #b = self.indices[-1]
    #t = 2(x-a)/(b-a) -1 scales x in [a,b] to [-1,1]
    #Fit becomes y = alpha1 t + alpha2
    #alpha1 = k1 *a
    #alpha2 = k1 *(b+a)/2 + k2
    #truefit  y = k1 x + k2
    # scaled_indices = 2*self.indices/b -1

    ###         Note numpy.polynomial.Polynomial.fit offers built in scaling and shifting of data
    ###                                 more numerically stable
    new_data = []
    new_indices = []
    for bound in data_bounds:
        new_data.extend(data[bound[0]:bound[1]])
        new_indices.extend(indices[bound[0]:bound[1]])
    return poly.fit(new_indices,new_data,1)

class data:
    def __init__(self, filename, BeatRunAvgN=100, beatnote_det_f=80, file_skip_lines=0):
        pure_dat = simple_dat_get(filename, file_skip_lines)
        Tavg = pure_dat[:,0:int(pure_dat.shape[1]/2)-1].mean(1)
        Pavg = pure_dat[:,int(pure_dat.shape[1]/2)-1:pure_dat.shape[1]-2].mean(1)
        ogbeat = pure_dat[:,pure_dat.shape[1]-2]
        self.indices = pure_dat[:,pure_dat.shape[1]-1]
        self.indices = self.indices - self.indices.min()
        self.scaledT = Tavg/Pavg
        

        #beatnoteClean
        (peak_indices,peak_val,peak_indices2,peak_val2, filteredBeat) = process_beatnote(self.indices,ogbeat,BeatRunAvgN)
        (cleared_indices, cleared_peaks) = cutoff_ends(peak_indices2.copy(), peak_val2.copy(), 2340, len(self.indices))
        differences = list(map(lambda x1,x2:x1-x2, cleared_indices[1:], cleared_indices[:len(cleared_indices)-1]))
        avg_diff = np.mean(differences)
        freq = [0]
        for diff in differences:
            if diff < avg_diff:
                freq.append(freq[-1] + beatnote_det_f)
            else:
                freq.append(freq[-1] + 250 - beatnote_det_f)
        
        # print(freq)
        beatnotefit = poly.fit(cleared_indices, freq,[0,1,2,3])

        #894 clean
        scan894 = True
        if scan894:
            data_bounds = [[0,2600],[6000,len(self.indices)]]
            coeff = LinFit(data_bounds, self.indices, self.scaledT)
            # print(coeff(self.indices))
        

        print(set(peak_indices2).issuperset(set(peak_indices)))
        plt.plot(self.indices,self.scaledT)
        plt.plot(self.indices,coeff(self.indices),'-r')
        plt.show()
        plt.plot(self.indices, coeff(self.indices)-self.scaledT)
        plt.show()
        plt.scatter(self.indices, ogbeat)
        plt.scatter(peak_indices,peak_val, color='red', marker='x')
        plt.show()
        plt.plot(self.indices, filteredBeat,linewidth=0.5,marker='.', mew='0.05')
        plt.scatter(peak_indices2,peak_val2,color='red', marker='x')
        plt.xlim(0,len(self.indices))
        plt.ylim(0, max(peak_val2)+0.1)
        temp = np.std(filteredBeat[BeatRunAvgN+1:len(self.indices)-BeatRunAvgN-1])*2
        plt.axvspan(0,1000,alpha=0.2,color='grey')
        plt.plot([0,len(self.indices)],[temp, temp], '-r')
        plt.show()
        temp = np.array(freq) - beatnotefit(np.array(cleared_indices))
        plt.scatter(cleared_indices,temp)
        plt.show()
        plt.plot(beatnotefit.linspace(1000)[0],beatnotefit.linspace(1000)[1],'-r')
        plt.scatter(cleared_indices,freq)
        plt.show()

        

# data(r"D:\Diego\git\6s7p\456Scan.csv", 100)
data(r"C:\Users\wolfw\Downloads\BeatnoteProcess7-321-25\894Scan.csv", beatnote_det_f=50 )
        


        
