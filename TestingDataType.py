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
    def __init__(self, par_folder,BeatRunAvgN=100, beatnote_det_f=0, beat_rng=[2340,8000], back_rngs=[[0,0],[6300,8000]], file_skip_lines=0, scan='456', exists = False):
        if not exists:
            if scan == '456':
                filename = par_folder + r"\456Scan.csv"
            else:
                filename = par_folder + r"\894Scan.csv"

            if beatnote_det_f == 0:
                if scan == '456':
                    beatnote_det_f = 40
                else:
                    beatnote_det_f = 20
            self.beatnote_det_f = beatnote_det_f
            self.BeatRunAvgN = BeatRunAvgN
            pure_dat = simple_dat_get(filename, file_skip_lines)
            Tavg = pure_dat[:,0:int(pure_dat.shape[1]/2)-1].mean(1)
            Pavg = pure_dat[:,int(pure_dat.shape[1]/2)-1:pure_dat.shape[1]-2].mean(1)
            ogbeat = pure_dat[:,pure_dat.shape[1]-2]
            self.indices = pure_dat[:,pure_dat.shape[1]-1]
            self.indices = self.indices - self.indices.min()
            self.scaledT = Tavg/Pavg
            self.scan = scan
            self.back_rngs = back_rngs
            self.beat_rng = beat_rng
            
            if scan == '456':
                folder = par_folder + r'\Analysis\456'
            else:
                folder = par_folder + r'\Analysis\894'
            self.folder = folder
            #Transition clean up 
            self.calculate_T_shift() 
            (peak_indices,peak_val,self.peak_indices2,self.peak_val2, self.filteredBeat) = process_beatnote(self.indices,ogbeat,BeatRunAvgN)
            #beatnoteClean
            self.calculate_beat_fit()

            np.savetxt(folder+r'\indices.csv', self.indices, fmt='%i', delimiter=',')
            np.savetxt(folder+r'\fitting\original\Tavg.csv', Tavg, delimiter=',')
            np.savetxt(folder+r'\fitting\original\Pavg.csv', Pavg, delimiter=',')
            np.savetxt(folder+r'\fitting\processed\scaledT.csv', self.scaledT, delimiter=',')
            np.savetxt(folder+r'\beatnote\original\ogbeat.csv', ogbeat, delimiter=',')
            np.savetxt(folder+r'\beatnote\original\peak_indices.csv', peak_indices, fmt='%i', delimiter=',')
            np.savetxt(folder+r'\beatnote\original\peak_val.csv', peak_val,  delimiter=',')
            np.savetxt(folder+r'\beatnote\processed\filteredBeat.csv', self.filteredBeat, delimiter=',')
            np.savetxt(folder+r'\beatnote\processed\peak_indices.csv', self.peak_indices2, fmt='%i', delimiter=',')
            np.savetxt(folder+r'\beatnote\processed\peak_val.csv', self.peak_val2, delimiter=',')

            #filter out peaks from og data at ends to make comparable
            (peak_indices, peak_val) = cutoff_ends(peak_indices, peak_val, BeatRunAvgN, len(self.indices)-BeatRunAvgN)
            self.beatselect_good = set(self.peak_indices2).issuperset(set(peak_indices))             

            plt.plot(self.indices,Tavg)
            plt.title('Averaged Transmission Measurements')
            plt.savefig(folder+r'\plots\Tavg.png')
            plt.clf()
            # plt.show()
            plt.plot(self.indices,Pavg)
            plt.title(r'Averaged Laser Power Measurements')
            plt.savefig(folder+r'\plots\Pavg.png')
            plt.clf()

            # plt.show()
            plt.scatter(self.indices, ogbeat)
            plt.title('Original Beanote')
            plt.scatter(peak_indices,peak_val, color='red', marker='x')
            plt.savefig(folder+r'\plots\ogbeat.png')
            plt.clf()
            # plt.show()

        else:
            self.scan = scan
            if scan == '456':
                folder = par_folder + r'\Analysis\456'
            else:
                folder = par_folder + r'\Analysis\894'
            if beatnote_det_f == 0:
                if scan == '456':
                    beatnote_det_f = 40
                else:
                    beatnote_det_f = 20
            self.beatnote_det_f = beatnote_det_f
            self.folder = folder
            self.indices = np.loadtxt(folder+r'\indices.csv', dtype=int, delimiter=',')
            self.scaledT = np.loadtxt(folder+r'\fitting\processed\scaledT.csv', delimiter=',')
            self.filteredBeat = np.loadtxt(folder+r'\beatnote\processed\filteredBeat.csv', delimiter=',')
            self.peak_indices2 = np.loadtxt(folder+r'\beatnote\processed\peak_indices.csv', dtype=int, delimiter=',').tolist()
            self.peak_val2 = np.loadtxt(folder+r'\beatnote\processed\peak_val.csv', delimiter=',').tolist()
            self.BeatRunAvgN = BeatRunAvgN

            temp = np.loadtxt(folder+r'\beatnote\processed\beat_fit_param.csv', delimiter=',') 
            #Save as domain, window, coef
            self.beatfit = poly(temp[4:], temp[0:2], temp[2:4])

            temp = np.loadtxt(folder+r'\beatnote\processed\beat_fit_param.csv', delimiter=',') 
            self.backgoundFit = poly(temp[4:], temp[0:2], temp[2:4])

            self.scaled_residuals = np.loadtxt(folder+r'\beatnote\processed\scaled_residuals.csv', delimiter=',')
            self.beat_rng = np.loadtxt(folder+r'\entries\beat_rng.csv', dtype=int, delimiter=',').tolist()
            temp = np.loadtxt(folder+r'\entries\back_rngs.csv', dtype=int, delimiter=',').tolist()
            self.back_rngs = [temp[:2],temp[2:]]
            (self.cleared_indices, self.cleared_peaks) = cutoff_ends(self.peak_indices2.copy(), self.peak_val2.copy(), self.beat_rng[0], self.beat_rng[1])
    
    def calculate_T_shift(self):
        self.backgoundFit = LinFit(self.back_rngs, self.indices, self.scaledT)
        back_fit_param = self.backgoundFit.domain.tolist()
        back_fit_param.extend(self.backgoundFit.window.tolist())
        back_fit_param.extend(self.backgoundFit.coef.tolist())
        np.savetxt(self.folder+r'\fitting\processed\back_fit_param.csv', back_fit_param, delimiter=',')

        temp = [self.back_rngs[0][0],self.back_rngs[0][1], self.back_rngs[1][0],self.back_rngs[1][1]]

        np.savetxt(self.folder+r'\entries\back_rngs.csv', temp, fmt='%i', delimiter=',')

        plt.plot(self.indices,self.scaledT)
        plt.plot(self.indices,self.backgoundFit(self.indices),'-r')
        plt.title(r'\frac{Transmission}{Laser Power}')
        plt.savefig(self.folder+r'\plots\scaledT_and_fit.png')
        plt.clf()
        # plt.show()
        self.correctedT = self.scaledT - self.backgoundFit(self.indices)
        plt.plot(self.indices, self.correctedT)
        plt.title('Background Corrected Transmission')
        plt.savefig(self.folder+r'\plots\correctedT.png')
        plt.clf()
            # plt.show()
    
    def calculate_beat_fit(self):
        (self.cleared_indices, self.cleared_peaks) = cutoff_ends(self.peak_indices2.copy(), self.peak_val2.copy(), self.beat_rng[0], self.beat_rng[1])
        self.differences = list(map(lambda x1,x2:x1-x2, self.cleared_indices[1:], self.cleared_indices[:len(self.cleared_indices)-1]))
        avg_diff = np.mean(self.differences)
        self.freq = [0]
        for diff in self.differences:
            if diff < avg_diff:
                self.freq.append(self.freq[-1] + self.beatnote_det_f*2)
            else:
                self.freq.append(self.freq[-1] + 250 - self.beatnote_det_f*2)
        # print(self.freq)

        
        self.beatfit = poly.fit(self.cleared_indices, self.freq,[0,1,2,3])
        beat_fit_param = self.beatfit.domain.tolist()
        beat_fit_param.extend(self.beatfit.window.tolist())
        beat_fit_param.extend(self.beatfit.coef.tolist())
        
        np.savetxt(self.folder+r'\entries\beat_rng.csv', self.beat_rng, fmt='%i', delimiter=',')
        np.savetxt(self.folder+r'\beatnote\processed\beat_fit_param.csv', beat_fit_param, delimiter=',')
        self.resid = self.freq-self.beatfit(np.array(self.cleared_indices))
        self.scaled_residuals = self.resid**2
        self.scaled_residuals[1:] = self.scaled_residuals[1:]/self.freq[1:]
        np.savetxt(self.folder+r'\beatnote\processed\scaled_residuals.csv', self.scaled_residuals, delimiter=',')
        print(self.scaled_residuals[1:].mean())
        plt.scatter(self.cleared_indices[1:],self.scaled_residuals[1:])
        plt.title(self.scan + 'Scaled Residuals, Mean='+str(np.mean(self.scaled_residuals[1:])))
        # plt.show()
        plt.savefig(self.folder+r'\plots\ScaledResiduals.png')
        plt.clf()

        plt.title(self.scan+' Filtered Beatnote with identified Peaks')
        plt.plot(self.indices, self.filteredBeat,linewidth=0.5,marker='.', mew='0.05')
        plt.scatter(self.peak_indices2,self.peak_val2,color='red', marker='x')
        plt.xlim(0,len(self.indices))
        plt.ylim(0, max(self.peak_val2)+0.1)
        temp = np.std(self.filteredBeat[self.BeatRunAvgN+1:len(self.indices)-self.BeatRunAvgN-1])*2
        plt.axvspan(0,self.beat_rng[0],alpha=0.2,color='grey')
        plt.axvspan(self.beat_rng[1],len(self.indices),alpha=0.2,color='grey')
        plt.plot([0,len(self.indices)],[temp, temp], '-r')
        plt.savefig(self.folder+r'\plots\filteredbeat.png')
        plt.clf()
        # plt.show()
        plt.title(self.scan+' Beatnote Fitting')
        plt.plot(self.beatfit.linspace(1000)[0],self.beatfit.linspace(1000)[1],'-r')
        plt.scatter(self.cleared_indices,self.freq)
        plt.savefig(self.folder+r'\plots\fitted_beat.png')
        plt.clf()
        # plt.show()
        temp = np.array(self.freq) - self.beatfit(np.array(self.cleared_indices))
        plt.scatter(self.cleared_indices,temp)
        plt.title(self.scan+' Beat Unscaled Residuals')
        plt.savefig(self.folder+r'\plots\unscaledresiduals.png')
        plt.clf()


            
        

# dat = data(r"D:\Diego\git\6s7p\Aug01,2025+4-20-28PM", 80, exists=False)
# print(dat.indices)
# print(dat.scaledT)




        
