import numpy as np
from numpy.polynomial import Polynomial as poly
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from scipy.special import wofz as wofz
from scipy.integrate import quad
import scipy as sci
from scipy import optimize as opt
from matplotlib import lines as lines
from numpy import pi as pi
import os as os
import lmfit as lm 

s2pi = np.sqrt(2*pi)
s2 = np.sqrt(2.0)

def voigt(x, amplitude=1.0, center=0.0, sigma=1.0, gamma=None):
    """Return a 1-dimensional Voigt function.

    voigt(x, amplitude, center, sigma, gamma) =
        amplitude*real(wofz(z)) / (sigma*s2pi)

    For more information, see: https://en.wikipedia.org/wiki/Voigt_profile

    """
    if gamma is None:
        gamma = sigma
    z = (x-center + 1j*gamma) / (sigma*s2)
    return amplitude*np.real(wofz(z)) / (sigma*s2pi)

def convertAD590(V):
    return V*20

def vapor_pres(T):
    #T in Celsius
    if T < 28.5:
        T =T+273.15
        Pres = 2.881+4.711-3999/T
    else:
        T =T+273.15
        Pres = 2.881+4.165-3830/T
    return 10**(Pres)

def delta(x):
    return list(map(lambda m,n:m-n,x[1:],x))

def get_frequency_steps(peaks, det_freq):
    #input of peak indices as list
    #assumption of any missing peak has at least 2 good peak pairs in between bad/missing peaks

    #returns (frequency at peaks, frequency diff between peaks, bad peaks, bad peak types)
    dX = delta(peaks)
    avg = np.mean(dX)
    good = []
    good_rights=[]
    freq = [0]
    freq_diff = []
    w0 = 0.250
    w1 = 2*det_freq
    dw = w0 - w1
    for i, dx in enumerate(dX):
        freq_diff.append(0)
        if dx < avg:
            good.append(i)
            freq_diff[-1]= w1
            good_rights.append(i+1)
            good.append(i+1)
        else:
            if len(freq_diff)>1:
                if freq_diff[-2] != 0:
                    freq_diff[-1] = dw
    og_set = np.arange(0,len(peaks),1).tolist()
    bad = list(set(og_set).difference(set(good)))
    bad.sort()
    double_bad = []
    if len(bad) !=0:
        bad_peak_type = []
        for i in range(len(bad)):
            #-1 is begining of peaks #2 is last of peaks
            #0 is only left peak, 1 is only right peak
            if bad[i] == 0:
                bad_peak_type.append(-1)
            elif bad[i] == len(peaks)-1:
                bad_peak_type.append(2)
            else:
                if bad[i] != bad[-1]:
                    if bad[i+1] != bad[i]+1 and bad[i-1] !=bad[i]-1:
                        if dX[bad[i]-1] < dX[bad[i]]:
                            #Then peak we have is correct distance from the previous peak so left peak
                            bad_peak_type.append(0)
                            freq_diff[bad[i]] =w0
                        else:
                            #Then peak is correct distance from next peak
                            if bad[i] != bad[i-1]+1:
                                bad_peak_type.append(1)
                                if bad[i]-1 in good_rights:
                                    good_rights.remove(bad[i]-1)
                                good_rights.append(bad[i])
                                freq_diff[bad[i]-1] = w0
                                freq_diff[bad[i]] = dw
                    elif bad[i+1] == bad[i]+1 and bad[i+1] != len(dX):
                        double_bad.append((bad[i],bad[i+1]))
                else:
                    if dX[bad[i]-1] < dX[bad[i]]:
                        #Then peak we have is correct distance from the previous peak so left peak
                        bad_peak_type.append(0)
                        freq_diff[bad[i]] =w0
                    else:
                        #Then peak is correct distance from next peak
                        if bad[i] != bad[i-1]+1:
                            bad_peak_type.append(1)
                            if bad[i]-1 in good_rights:
                                good_rights.remove(bad[i]-1)
                            good_rights.append(bad[i])
                            freq_diff[bad[i]-1] = w0
                            freq_diff[bad[i]] = dw
        
        if (bad[0] == 0) or (bad[-1] == len(peaks)-1):
            track=[[0,0],[0,0],[0,0]]
            #tracking large gap averages on left and right ends of beatnote data so right peak
            for i in good_rights:
                if i < len(peaks)*0.25:
                    track[0][0]+= dX[i]
                    track[1][0]+=1
                elif i > len(peaks) *0.65 and i<len(peaks)*0.9:
                    #exclude last one as could be corrupted by being too far from last peak
                    track[0][1]+=dX[i]
                    track[1][1]+=1
            track[2][0] = track[0][0]/track[1][0]
            track[2][1] = track[0][1]/track[1][1]
            # print(bad)
            if bad[0] == 0:
                if dX[0] < track[2][0]*1.15:
                    #then its the right peak because correct distance from next peak pair
                    bad_peak_type[0] = 1
                    freq_diff[0] = dw
                else:
                    bad_peak_type[0] = 0
                    freq_diff[0] = w0
            if bad[-1] == len(peaks)-1:
                if dX[-1] < track[2][1]*1.15:
                    #then its the left peak because correct distance from last peak pair
                    bad_peak_type[0] = 0
                    freq_diff[-1] = dw
                else:
                    bad_peak_type[0] = 1
                    freq_diff[-1] = w0
        if len(double_bad) != 0:
            freq_fix = [[dw,w0,w0],[dw,w0+w1,dw],[w0,dw,w0],[w0,w0,dw]]
            for h in double_bad:
                #4 opitions
                #[LL,LR,RL,RR] -> [freq_diff[h[0]-1],freq_diff[h[0],freq_diff[h[1]]]
                if abs(dX[h[0]]-dX[h[1]]) < dX[h[0]] *0.15: #For explanation refer to notes, case 1
                    freq_diff[h[0]-1] = freq_fix[0][0]
                    freq_diff[h[0]] = freq_fix[0][1]
                    freq_diff[h[1]] = freq_fix[0][2]
                else:
                    if abs(dX[h[0]-1]-dX[h[0]]) < dX[h[0]-1] *0.15:#Case 4
                        freq_diff[h[0]-1] = freq_fix[3][0]
                        freq_diff[h[0]] = freq_fix[3][1]
                        freq_diff[h[1]] = freq_fix[3][2]
                    else:
                        if dX[h[0]] > dX[h[1]]: #case 2
                            freq_diff[h[0]-1] = freq_fix[1][0]
                            freq_diff[h[0]] = freq_fix[1][1]
                            freq_diff[h[1]] = freq_fix[1][2]
                        else:#case 3
                            freq_diff[h[0]-1] = freq_fix[2][0]
                            freq_diff[h[0]] = freq_fix[2][1]
                            freq_diff[h[1]] = freq_fix[2][2]

    else:
        bad = []
        bad_peak_type = []
    for i in range(len(dX)):
        freq.append(freq[-1]+freq_diff[i])
    
    return (freq, freq_diff, bad, bad_peak_type)

    # # freq_diff 
    # if bad[0] == 0:
    #     if bad_peak_type[0] ==1:
    #         freq_diff[0] = 0.250 - 2*det_freq
    #     else:
    #         freq_diff[0] = 0.250
    # if bad[-1] == len(peaks)-1:
    #     if dX[-1] < track[2][1]:
            

def simple_dat_get(filename, skip_lines=0):
    file = open(filename, 'r')
    data = []
    for i in range(skip_lines):
        file.readline()
    for line in file:
        x = list(map(float, line.strip().split(',')))
        data.append(x)
    # print(data)
    file.close()
    return np.array(data)

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
    upper = np.mean(data[data_bounds[1][0]:data_bounds[1][1]])-np.mean(data[data_bounds[0][0]:data_bounds[0][1]])
    lower = np.mean(indices[data_bounds[1][0]:data_bounds[1][1]])-np.mean(indices[data_bounds[0][0]:data_bounds[0][1]])
    fitted_param, pcov = opt.curve_fit(lambda x,k,b:k*x+b, new_indices,new_data,[upper/lower,new_data[0]])
    return fitted_param

def LinFit2(data_bounds, indices, data):
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
    upper = np.mean(data[data_bounds[1][0]:data_bounds[1][1]])-np.mean(data[data_bounds[0][0]:data_bounds[0][1]])
    lower = np.mean(indices[data_bounds[1][0]:data_bounds[1][1]])-np.mean(indices[data_bounds[0][0]:data_bounds[0][1]])
    fitted_param, pcov = opt.curve_fit(lambda x,k2,k,b:k2*x**2 + k*x+b, new_indices,new_data,[0.01,upper/lower,new_data[0]])
    return fitted_param

def LinFit3(data_bounds, indices, data):
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
    # upper = np.mean(data[data_bounds[1][0]:data_bounds[1][1]])-np.mean(data[data_bounds[0][0]:data_bounds[0][1]])
    # lower = np.mean(indices[data_bounds[1][0]:data_bounds[1][1]])-np.mean(indices[data_bounds[0][0]:data_bounds[0][1]])
    fitted_param, pcov = opt.curve_fit(lambda x,p0,shift,k:k*(x-shift)**2+p0, new_indices,new_data,[max(new_data),1,-0.001])
    return fitted_param



def cvt_abs_wav_to_diff(abs_wav):
    #assume units of wavnum are 1/cm
    c=299792458
    x = [0]
    for i in range(len(abs_wav)-1):
        x.append((abs_wav[i+1]-abs_wav[i])*c/10000000)
    return x #returns GHz freq diff


class data:
    def __init__(self, par_folder,BeatRunAvgN=100, scan='456', exists = False):
        self.scan = scan
        self.par_folder = par_folder
        self.BeatRunAvgN = BeatRunAvgN
        self.beat_height = 0

        temp = np.loadtxt(self.par_folder+'\\beatnote_det_f.csv',delimiter=',')
        if scan == '456':
            self.folder = par_folder + r'\Analysis\456'
            self.BeatRunAvgN=75
            self.beatnote_det_f=temp[0]
        else:
            self.folder = par_folder + r'\Analysis\894'
            self.BeatRunAvgN=100
            self.beatnote_det_f=temp[1]
        if not exists:
            self.beat_rng = [0,8000]
            self.make_init_plots()
        else:
            self.load_dat()

        if scan == '894':
            peaks, properties = find_peaks(-self.scaledT,width=500, prominence=0.1)
            if (properties['right_ips'][1] - properties['left_ips'][1]) > (properties['right_ips'][0] - properties['left_ips'][0]):
                            #peak 2 is larger 
                            self.set_transition(F1=3)
                            self.F1 = 3
            else:
                #peak 1 is larger
                self.set_transition(F1=4)
                self.F1 = 4
        else:
            self.F1 = 3
            self.set_transition(F1=3)
        temp = np.loadtxt(par_folder+'\\TemperatureV2.csv').mean(0)
        if self.scan == '456':
            self.coldfinger = temp[0:2].mean()
            self.hotbody = temp[3:5].mean()
        else:
            self.coldfinger = temp[1:3].mean()
            self.hotbody = temp[4:6].mean()

    def make_init_plots(self):
        self.beat_height = np.savetxt(self.folder+'\\entries\\beat_peak_min.csv',[0],delimiter=',')
        pure_dat = np.loadtxt(self.par_folder+'\\'+self.scan+'.csv', delimiter=',')
        background = np.loadtxt(self.par_folder+"\\Background.csv",delimiter=',').mean(0)
        shape = pure_dat.shape[1]
        n = int((shape-2)/3)
        self.indices = pure_dat[:,shape-1]
        self.indices = self.indices - self.indices.min()
        np.savetxt(self.folder+r'\indices.csv', self.indices, fmt='%i', delimiter=',')
        self.indices=self.indices.tolist()

        Tavg = pure_dat[:,:n].mean(1)
        Pavg = pure_dat[:,n:2*n].mean(1)
        Havg = pure_dat[:,2*n:3*n].mean(1)

        plt.plot(self.indices, Tavg)
        plt.plot([self.indices[0],self.indices[-1]],[background[0],background[0]],'-r')
        plt.title(r'Averaged Transmission Measurements')
        plt.savefig(self.folder+r'\plots\Tavg.png')
        plt.clf()

        plt.plot(self.indices, Pavg)
        plt.plot([self.indices[0],self.indices[-1]],[background[1],background[1]],'-r')
        plt.title(r'Averaged Laser Power Measurements')
        plt.savefig(self.folder+r'\plots\Pavg.png')
        plt.clf()

        plt.plot(self.indices, Havg)
        plt.plot([self.indices[0],self.indices[-1]],[background[2],background[2]],'-r')
        plt.title(r'Averaged Hot Cell Measurements')
        plt.savefig(self.folder+r'\plots\Havg.png')
        plt.clf()

        self.scaledT = (Tavg-background[0])/(Pavg-background[1])
        self.scaledH = (Havg-background[2])/(Pavg-background[1])
        
        np.savetxt(self.folder+r'\fitting\processed\scaledT.csv', self.scaledT, delimiter=',')
        np.savetxt(self.folder+r'\fitting\processed\scaledH.csv', self.scaledH, delimiter=',')

        plt.plot(self.indices, self.scaledT)
        plt.title(r'Scaled Transmission')
        plt.savefig(self.folder+r'\plots\Pavg.png')
        plt.clf()

        plt.plot(self.indices, self.scaledH)
        plt.plot([self.indices[0],self.indices[-1]],[background[2],background[2]],'-r')
        plt.title(r'Scaled Hot Cell')
        plt.savefig(self.folder+r'\plots\Havg.png')
        plt.clf()

        ogbeat = pure_dat[:,n-2]
        self.init_process_beatnote(self.indices,ogbeat,self.BeatRunAvgN)

    def init_process_beatnote(self,indices,ogBeat,run_avg_num):
        #Initial Peak Finding with base data
        peak_indices = find_peaks(ogBeat, height=-50, distance=50)
        peak_val = peak_indices[1]['peak_heights']
        peak_indices = peak_indices[0]
        #filtering data
        runningavg = np.convolve(ogBeat, np.ones(run_avg_num)/run_avg_num, mode='same')
        self.filteredBeat=ogBeat-runningavg
        #Peak finding filtered data
        standard_peak_min = np.std(self.filteredBeat[run_avg_num+1:len(indices)-run_avg_num-1])*2
        peak_indices2 = find_peaks(self.filteredBeat, height=standard_peak_min,distance=20)
        peak_indices = peak_indices.tolist()
        peak_val = peak_val.tolist()
        peak_val2 = peak_indices2[1]['peak_heights']
        peak_indices2 = peak_indices2[0]
        plt.plot(indices, ogBeat,linewidth=0.5,marker='.', mew='0.05')
        plt.title('Original Beanote')
        plt.scatter(peak_indices,peak_val, color='red', marker='x')
        plt.savefig(self.folder+r'\plots\ogbeat.png')
        plt.clf()
        (self.peak_indices, self.peak_val) = correct_peaks(peak_indices=peak_indices2.tolist(), peak_val=peak_val2.tolist())
        np.savetxt(self.folder+r'\beatnote\processed\filteredBeat.csv', self.filteredBeat, delimiter=',')
        np.savetxt(self.folder+r'\beatnote\processed\peak_indices.csv', self.peak_indices, fmt='%i', delimiter=',')
        np.savetxt(self.folder+r'\beatnote\processed\peak_val.csv', self.peak_val, delimiter=',')
        self.beat_height = 0
        self.filter_beatnote()

    def filter_beatnote(self):
        self.beat_height = np.loadtxt(self.folder+'\\entries\\beat_peak_min.csv',delimiter=',')[0]
        if self.beat_height == 0:
            standard_peak_min = np.std(self.filteredBeat[self.BeatRunAvgN+1:len(self.indices)-self.BeatRunAvgN-1])*2
        else:
            standard_peak_min = self.beat_height
        peak_indices2 = find_peaks(self.filteredBeat, height=standard_peak_min,distance=20)
        peak_val2 = peak_indices2[1]['peak_heights']
        peak_indices2 = peak_indices2[0]
        # (peak_indices2, peak_val2) = correct_peaks(peak_indices=peak_indices2.tolist(), peak_val=peak_val2.tolist())

        self.peak_indices = peak_indices2
        self.peak_val = peak_val2
        np.savetxt(self.folder+r'\beatnote\processed\peak_indices.csv',self.peak_indices, delimiter=',')
        np.savetxt(self.folder+r'\beatnote\processed\peak_val.csv',self.peak_val, delimiter=',')
        if not type(self.peak_indices) == list:
            (self.cleared_indices, self.cleared_peaks) = cutoff_ends(self.peak_indices.copy().tolist(), self.peak_val.copy().tolist(), self.beat_rng[0], self.beat_rng[1])
        else:(self.cleared_indices, self.cleared_peaks) = cutoff_ends(self.peak_indices.copy(), self.peak_val.copy(), self.beat_rng[0], self.beat_rng[1])
        
        plt.title(self.scan+' Filtered Beatnote with identified Peaks')
        plt.plot(self.indices, self.filteredBeat,linewidth=0.5,marker='.', mew='0.05')
        plt.scatter(self.peak_indices,self.peak_val,color='red', marker='x')
        plt.xlim(0,len(self.indices))
        if max(self.peak_val)<25:
            plt.ylim(0, max(self.peak_val)+0.1)
        else:
            plt.ylim(0, np.std(self.peak_val)*2)
        if self.beat_height == 0:
            temp = np.std(self.filteredBeat[self.BeatRunAvgN+1:len(self.indices)-self.BeatRunAvgN-1])*2
        else:
            temp = self.beat_height
        plt.axvspan(0,self.beat_rng[0],alpha=0.2,color='grey')
        plt.axvspan(self.beat_rng[1],len(self.indices),alpha=0.2,color='grey')
        plt.plot([0,len(self.indices)],[temp, temp], '-r')
        plt.savefig(self.folder+r'\plots\filteredbeat.png')
        plt.clf()

    def loaddat(self):
        self.indices = np.loadtxt(self.folder+r'\indices.csv', dtype=int, delimiter=',')
        self.scaledT = np.loadtxt(self.folder+r'\fitting\processed\scaledT.csv', delimiter=',')
        self.scaledH = np.loadtxt(self.folder+r'\fitting\processed\scaledH.csv', delimiter=',')
        self.filteredBeat = np.loadtxt(self.folder+r'\beatnote\processed\filteredBeat.csv', delimiter=',')
        self.peak_indices = np.loadtxt(self.folder+r'\beatnote\processed\peak_indices.csv', dtype=int, delimiter=',').tolist()
        self.peak_val = np.loadtxt(self.folder+r'\beatnote\processed\peak_val.csv', delimiter=',').tolist()

    def find_pairs(self, peak_indices, peak_val):
        differences = list(map(lambda x1,x2:x1-x2, self.cleared_indices[1:], self.cleared_indices[:len(self.cleared_indices)-1]))
        close_pairs = []
        far_pairs = []
        bads_close = []
        bads_far = []
        avg_diff = np.mean(differences)
        pattern = -1
        for i,diff in enumerate(differences):
            if diff< i:
                if pattern == -1:
                    pattern = 0
                    start=0
                elif pattern == 1:
                    #good switch
                    pattern = 0
                else:
                    bads_close.append([i-1,i,i+1])
                close_pairs.append([i, i+1])
            else:
                if pattern == -1:
                    pattern = 1
                    start = 1
                elif pattern == 0:
                    #good switch
                    pattern = 1
                    far_pairs.append([i,i+1])
                else:
                    bads_far.append([i-1,i,i+1])

    def calculate_beat_fit(self):
        if not type(self.peak_indices2) == list:
            (self.cleared_indices, self.cleared_peaks) = cutoff_ends(self.peak_indices2.copy().tolist(), self.peak_val2.copy().tolist(), self.beat_rng[0], self.beat_rng[1])
        else:(self.cleared_indices, self.cleared_peaks) = cutoff_ends(self.peak_indices2.copy(), self.peak_val2.copy(), self.beat_rng[0], self.beat_rng[1])
        # (self.cleared_indices, self.cleared_peaks)=(self.peak_indices2.copy(), self.peak_val2.copy())
        # self.differences = list(map(lambda x1,x2:x1-x2, self.cleared_indices[1:], self.cleared_indices[:len(self.cleared_indices)-1]))
        # avg_diff = np.mean(self.differences)
        # self.freq = [0]
        # for diff in self.differences:
        #     if diff < avg_diff:
        #         self.freq.append(self.freq[-1] + self.beatnote_det_f*2)
        #     else:
        #         self.freq.append(self.freq[-1] + 0.250 - self.beatnote_det_f*2)
        # print(self.freq)
        (self.freq, freq_diff,bad,bad_peak_type) = get_frequency_steps(self.cleared_indices,self.beatnote_det_f)
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
        if max(self.peak_val2)<25:
            plt.ylim(0, max(self.peak_val2)+0.1)
        else:
            plt.ylim(0, np.std(self.peak_val2)*2)
        if self.beat_height == 0:
            temp = np.std(self.filteredBeat[self.BeatRunAvgN+1:len(self.indices)-self.BeatRunAvgN-1])*2
        else:
            temp = self.beat_height
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

        plt.plot(self.beatfit(self.indices[self.beat_rng[0]:self.beat_rng[1]]),self.scaledT[self.beat_rng[0]:self.beat_rng[1]])
        plt.title(self.scan+' beat fit scaledT')
        plt.savefig(self.folder+r'\plots\beatScaledT.png')
        plt.clf()
        
        plt.plot(self.indices,self.scaledT)
        plt.title(r'$\frac{Transmission}{Laser Power}$')
        plt.axvspan(0,self.beat_rng[0],alpha=0.2,color='grey')
        plt.axvspan(self.beat_rng[1],len(self.indices),alpha=0.2,color='grey')
        plt.savefig(self.folder+r'\plots\scaledT.png')
        plt.clf()


    def set_transition(self, F1):
        #Can be used to verify i some fashion if need be of the coefficient
        # fine_structure = 0.00729735256
        # Lcell = 29.09 #cm
        # denom = (7+1) * (1+2) #(2I+1)(2J+1) #I=7/2, J=1/2 6s starting ground state

        #Voigt(x,s,g)= Re(w(z))/s \rad(2pi) , z = (x + i g)/(s\rad(2)) ; w = Faddeeva function
        if self.scan == '456':
            #In GHz
            if F1 == 3:
                abs_wavenum = [21946.56347,21946.56514,21946.56736]
                #from W. Williams, M. Herd, and W. Hawkins, Laser Phys. Lett. 15,095702 (2018).
                self.abs_freq = list(map(lambda x:  x*29.9792458,abs_wavenum))#29.9..... = c decimals for scaling to convert to GHz
                self.center = np.mean(self.abs_freq)
                self.hypsplit = list(map(lambda x: x-self.abs_freq[0],self.abs_freq)) 
                self.hyp_weights = [5/12,7/26,5/16]
                #Higher frequency because its closer ie. F=3->F=2,3,4
            else:
                #F1 == 4
                abs_wavenum = [21946.25850,21946.26072,21946.26349]
                self.abs_freq = list(map(lambda x:  x*29.9792458,abs_wavenum))#29.9..... = c decimals for scaling for scaling to convert to GHz
                self.center = np.mean(self.abs_freq)
                self.hypsplit = list(map(lambda x: x-self.abs_freq[0],self.abs_freq)) 
                self.hyp_weights = [7/48,7/16,5/16]
                #Lower frequency because father F=4->F=3,4,5
        elif self.scan == '894':
            #already in GHz
            main_tran = 335116.048807
            center_to6PF3 = 0.656820
            center_to6PF4 = 0.510860
            center_to6SF3 = 5.170855370625
            center_to6SF4 = 4.021776399375
            if F1 == 3:
                self.abs_freq= [main_tran+center_to6SF3-center_to6PF3,main_tran+center_to6SF3+center_to6PF4]
                self.center = np.mean(self.abs_freq)
                self.hyp_weights = [7/24,7/8]
                self.hypsplit = [0,1.167680]
                #Higher frequency because its closer ie. F=3->F=3,4
            else:
                #F1 == 4
                self.abs_freq = [main_tran-center_to6SF4-center_to6PF3,main_tran-center_to6SF4+center_to6PF4]
                self.center = np.mean(self.abs_freq)
                self.hyp_weights = [7/8,5/8]
                # self.hyp_weights = [5/8,7/8]
                self.hypsplit = [0,1.167680]
                #Lower frequency because father F=4->F=3,4 
    
    def set_fitting_function(self):
            #New function using lmfit
            #constants wihout powers
            c=2.99792458
            afs=7.29735256
            m=2.2069484567911638
            kB=1.3806503
            k1 = np.sqrt(kB/m/c**2) * 10**(-7) #to be used for delta _wD = w *k1 *sqrt(T)


            if self.scan == '456':
                peaks, properties = find_peaks(-self.scaledT,width=500,prominence=0.02)
                p0 = 0.37 #scaledT pwr at top
                Gamma = 1/137.54/2/(2*pi)#half of lifetime in GHz from "Measurement of the lifetimes of the 7p 2P3/2 and 7p 2P1/2 states of atomic cesium" -us

            else:
                peaks, properties = find_peaks(-self.scaledT,width=500, prominence=0.1)
                peaks[0] = int((properties['left_ips'][0]+properties['right_ips'][0])/2)
                peaks[1] = int((properties['left_ips'][1]+properties['right_ips'][1])/2)
                p0=0.2 #scaledT power at top
                Gamma = 1/34.791/2/(2*pi) #half of lifetime in GHz from Stek
            guess = self.beatfit(peaks[0]) #guess of frequency location of first peak relative to begin of fit
            coeff = self.hyp_weights
            
            plotting_freq = self.beatfit(np.array(self.indices[self.beat_rng[0]:self.beat_rng[1]]))
            plotting_scaledT = self.scaledT[self.indices[self.beat_rng[0]]:self.indices[self.beat_rng[1]]]
            weights1 = plotting_freq.copy()
            weights1 = weights1.tolist()
            dist = 1000
            # for i in range(len(weights1)):
            #     if i < dist:
            #         weights1[i] = np.std(plotting_scaledT[0:dist])
            #     elif i > len(weights1) - dist-1:
            #         weights1[i] = np.std(plotting_scaledT[len(weights1)-dist:len(weights1)-1])
            #     else:
            #         weights1[i] = np.std(plotting_scaledT[i-int(dist/2):i+int(dist/2)])
            # if self.etalon_ranges[0][1] != 0:
            #     test = LinFit(self.etalon_ranges, self.beatfit(self.indices), self.scaledT)
            # else:
            if self.scan == '456':
                #fitting slope of background
                #this is 456 scan
                test = LinFit([[self.beat_rng[0],peaks[0]-int(properties['widths'][0]*1.5)],[peaks[0]+int(properties['widths'][0]*1.5),self.beat_rng[1]]], self.beatfit(self.indices), self.scaledT)
                test2 = LinFit2([[self.beat_rng[0],peaks[0]-int(properties['widths'][0]*1.5)],[peaks[0]+int(properties['widths'][0]*1.5),self.beat_rng[1]]], self.beatfit(self.indices), self.scaledT)
                test3 = LinFit3([[self.beat_rng[0],peaks[0]-int(properties['widths'][0]*1.5)],[peaks[0]+int(properties['widths'][0]*1.5),self.beat_rng[1]]], self.beatfit(self.indices), self.scaledT)

                gauss1 = lambda x,peak,cen,s: peak * np.exp(-((x-cen)/s)**2/2) + 1
                weights1 = gauss1(plotting_freq,6,self.beatfit(peaks[0]),0.15)
                plt.plot(plotting_freq,weights1)
                plt.show()
            else:
                # print('left',peaks[0]-int(properties['widths'][0]),'right',peaks[1]+int(properties['widths'][1]))
                test = LinFit([[self.beat_rng[0],peaks[0]-int(properties['widths'][0])],[peaks[1]+int(properties['widths'][1]),self.beat_rng[1]]], self.beatfit(self.indices), self.scaledT)
            if self.scan == '894':
                if self.use_cur_bot:
                    # ceiling = np.mean(self.scaledT[self.back_rngs[0][0]:self.back_rngs[0][1]])
                    # floor = np.mean(self.scaledT[self.back_rngs[1][0]:self.back_rngs[1][1]])
                    # baseline = floor/ceiling
                    baseline = np.mean(self.scaledT[self.back_rngs[1][0]:self.back_rngs[1][1]])
                else:
                    if os.path.exists(self.par_folder+'\\PwrWings894.csv'):
                        hotcell= np.loadtxt(self.par_folder+'\\PwrWings894.csv', delimiter=',')
                        baseline = hotcell[1]
                    else:
                        baseline = 0.05*np.mean(self.scaledT[self.beat_rng[0]:peaks[0]-int(properties['widths'][0]*1.5)]) #estimate power in wings for 894
            else:
                if os.path.exists(self.par_folder+'\\PwrWings456.csv'):
                    hotcell= np.loadtxt(self.par_folder+'\\PwrWings456.csv', delimiter=',')
                    bottom = hotcell[1]
                    top = hotcell[0]
                    baseline = (test3[2]*(self.beatfit(int(self.indices[peaks[0]]))-test3[1])**2+test3[0])*bottom/top
                else:
                    if self.F1 == 4:
                        baseline = 0.15*np.mean(self.scaledT[self.beat_rng[0]:peaks[0]-int(properties['widths'][0]*1.5)]) #estimate 15% power in wings for 456
                    if self.F1 == 3:
                        baseline = 0.008*np.mean(self.scaledT[self.beat_rng[0]:peaks[0]-int(properties['widths'][0]*1.5)]) #estimate 0.8% power in wings for 456
            # print(self.hotbody)
            if self.hotbody == 30:
                low = 20
                high = None
            else:
                low = self.hotbody-2
                high = self.hotbody+2
            params = lm.Parameters()
            # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
            if self.scan == '456':
                params.add_many(
                    # ('a', 6, True, 0, None, None, None),
                    # ('p0', test[1]-baseline, True, 0.7*(test[1]-baseline), 1.3*(test[1]-baseline), None, None),
                    # ('h1', test[0], False, test[0]-abs(test[0])*0.2, test[0]+abs(test[0])*0.2, None, None),
                    # ('mv', guess, True, 0, 4, None, None),
                    # ('T', self.hotbody, True, low, high, None, None),
                    # ('gamma', Gamma*1.3, False, Gamma, Gamma*3, None, None),
                    # ('base', baseline, False, baseline*0.8, baseline*1.2, None, None))


                # params.add_many( #Uses test2
                #     ('a', 6, True, 0, None, None, None),
                #     ('p0', test2[2]-baseline, True, 0.7*(test2[2]-baseline), 1.3*(test2[2]-baseline), None, None),
                #     ('h1', test2[1], True, test2[1]-abs(test2[1])*0.2, test2[1]+abs(test2[1])*0.2, None, None),
                #     ('h2', test2[0], True, test2[0]-abs(test2[0])*0.2, test2[0]+abs(test2[0])*0.2, None, None),
                #     ('mv', guess, True, 0, 4, None, None),
                #     ('T', self.hotbody, True, low, high, None, None),
                #     ('gamma', Gamma*1.3, False, Gamma, Gamma*3, None, None),
                #     ('base', baseline, False, baseline*0.8, baseline*1.2, None, None))
                
                # params.add_many( #using Test 3
                    ('a', 6, True, 0, None, None, None),
                    ('p0', test3[0]-baseline, False, 0.7*(test3[0]-baseline), 1.3*(test3[0]-baseline), None, None),
                    ('h1', test3[1], False, test3[1]-abs(test3[1])*0.2, test3[1]+abs(test3[1])*0.2, None, None),
                    ('h2', test3[2], False, test3[2]-abs(test3[2])*0.2, test3[2]+abs(test3[2])*0.2, None, None),
                    ('mv', guess, True, 0, 4, None, None),
                    ('T', self.hotbody, True, low, high, None, None),
                    ('gamma', Gamma*1.3, True, Gamma, Gamma*3, None, None),
                    ('base', baseline, False, baseline*0.7, baseline*1.3, None, None))
                    
            else:
                params.add_many(
                    ('a', 6, True, 0, None, None, None),
                    ('p0', test[1]-baseline, True, 0.7*(test[1]-baseline), 1.3*(test[1]-baseline), None, None),
                    ('h1', test[0], True, test[1]-abs(test[1])*0.2, test[0]+abs(test[0])*0.2, None, None),
                    ('mv', guess, True, 0, 4, None, None),
                    ('T', self.hotbody, True, low, high, None, None),
                    ('gamma', Gamma*1.3, False, Gamma, Gamma*3, None, None),
                    ('base', baseline, False, baseline*0.8, baseline*1.2, None, None))
            if self.scan == '456':
                params['a'].set(value=0.35)
                # params['gamma'].set(vary=False)
                params['base'].set(vary=False)
                # fun1 = lambda w,a,p0,h1,mv,T,gamma,base: (p0+h1*w)*np.exp(-a*((w-mv+self.abs_freq[0])/10**6)*(voigt(w,coeff[0],mv,np.sqrt(T+273.15)*k1*self.abs_freq[0],gamma)+
                #                                                             voigt(w,coeff[1],mv+self.hypsplit[1],np.sqrt(T+273.15)*k1*self.abs_freq[1],gamma)+
                #                                                             voigt(w,coeff[2],mv+self.hypsplit[2],np.sqrt(T+273.15)*k1*self.abs_freq[2],gamma))) + base
                # mod = lm.Model(fun1,['w'],['a','p0','h1','mv','T','gamma','base'])
                # result = mod.fit(self.scaledT[self.beat_rng[0]:self.beat_rng[1]],params=params,w=plotting_freq,method='ampgo')

                # fun1 = lambda w,a,p0,h1,h2,mv,T,gamma,base: p0*(1+h1*w+h2*w**2)*np.exp(-a*((w-mv+self.abs_freq[0])/10**6)*(voigt(w,coeff[0],mv,np.sqrt(T+273.15)*k1*self.abs_freq[0],gamma)+
                #                                                             voigt(w,coeff[1],mv+self.hypsplit[1],np.sqrt(T+273.15)*k1*self.abs_freq[1],gamma)+
                #                                                             voigt(w,coeff[2],mv+self.hypsplit[2],np.sqrt(T+273.15)*k1*self.abs_freq[2],gamma))) + base
                # mod = lm.Model(fun1,['w'],['a','p0','h1','h2','mv','T','gamma','base'])
                # result = mod.fit(self.scaledT[self.beat_rng[0]:self.beat_rng[1]],params=params,w=plotting_freq,method='shgo')
                
                #Redone quadratic fitting
                fun1 = lambda w,a,p0,h1,h2,mv,T,gamma,base: (h2*(w-h1)**2+p0)*np.exp(-a*((w-mv+self.abs_freq[0])/10**6)*(voigt(w,coeff[0],mv,np.sqrt(T+273.15)*k1*self.abs_freq[0],gamma)+
                                                                            voigt(w,coeff[1],mv+self.hypsplit[1],np.sqrt(T+273.15)*k1*self.abs_freq[1],gamma)+
                                                                            voigt(w,coeff[2],mv+self.hypsplit[2],np.sqrt(T+273.15)*k1*self.abs_freq[2],gamma))) + base
                mod = lm.Model(fun1,['w'],['a','p0','h1','h2','mv','T','gamma','base'])
                result = mod.fit(self.scaledT[self.beat_rng[0]:self.beat_rng[1]],params=params,w=plotting_freq,method='ampgo',weights=weights1)
            else:
                # params['gamma'].set(vary=True)
                params['gamma'].set(value=Gamma)
                params['base'].set(vary=True)
                fun1 = lambda w,a,p0,h1,mv,T,gamma,base: (p0+h1*w)*np.exp(-a*((w-mv+self.abs_freq[0])/10**6)*(voigt(w,coeff[0],mv,np.sqrt(T+273.15)*k1*self.abs_freq[0],gamma)+
                                                                                                            voigt(w,coeff[1],mv+self.hypsplit[1],np.sqrt(T+273.15)*k1*self.abs_freq[1],gamma))) + base
                mod = lm.Model(fun1,['w'],['a','p0','h1','mv','T','gamma','base'])
                result = mod.fit(self.scaledT[self.beat_rng[0]:self.beat_rng[1]],params=params,w=plotting_freq,method='ampgo')
            
            print(result.fit_report())
            resid = result.residual
            self.fitted_param = list(map(lambda key:result.params[key].value,result.params.keys()))
            self.pcov = list(map(lambda key:result.params[key].stderr,result.params.keys()))
            # resid = self.fitting_eqn3(plotting_freq,*fitted_param3)-self.scaledT[self.beat_rng[0]:self.beat_rng[1]]
            # chi2 = resid**2
            self.alpha = self.fitted_param[0]
            self.alph_err=self.pcov[0]
            np.savetxt(self.folder+r'\fitting\processed\fitting_param.csv', self.fitted_param, delimiter=',')
            try:
                np.savetxt(self.folder+r'\fitting\processed\pcov.csv',self.pcov,delimiter=',')
            except:
                self.pcov = np.zeros(len(self.fitted_param)).tolist()
            plt.scatter(plotting_freq,self.scaledT[self.beat_rng[0]:self.beat_rng[1]])
            # plt.plot(plotting_freq,result.best_fit, '-r',linewidth=0.2,marker='.')#, mew='0.05')
            plt.title(self.scan+ 'Fitted plot, a='+str(self.fitted_param[0]) + r', err='+ str(self.alph_err))
            plt.xlabel('Freq [GHz]')
            # plt.show()
            if self.scan == '456':
                # plt.plot(plotting_freq,test2[0]*(plotting_freq**2)+test2[1]*(plotting_freq)+test2[2],'-g')
                plt.plot(plotting_freq,fun1(plotting_freq,*self.fitted_param),'-r')
            else:
                plt.plot(plotting_freq,test[0]*(plotting_freq)+test[1],'-g')
                plt.plot(plotting_freq,fun1(plotting_freq,*self.fitted_param),'-r')
                # print(self.fitted_param)

            k3 =self.beatfit(properties["left_ips"])
            k4 = self.beatfit(properties["right_ips"])
            plt.vlines(x=self.beatfit(peaks), ymin= self.scaledT[peaks], ymax = properties['prominences']+self.scaledT[peaks], color = "blue")
            plt.hlines(y=-properties["width_heights"], xmin=k3,xmax=k4, color = "blue")
            # plt.plot(plotting_freq,weights1)
            plt.savefig(self.folder+r'\plots\FittedScan.png')
            plt.show()
            plt.clf()

            plt.scatter(plotting_freq,resid)
            # # plt.show()
            plt.title(self.scan+ 'Fitted plot residuals')
            plt.xlabel('Freq [GHz]')
            plt.savefig(self.folder+r'\plots\FittedScanResid.png')
            plt.clf()
            self.fitted = True
            if self.scan == '456':
                lines = {}
                date = self.par_folder[self.par_folder.rfind('/')+1:]
                temp = list(map(str, self.fitted_param))
                data = [date]
                data.extend(temp)
                day_path = self.par_folder[:self.par_folder.rfind('/')]
                fits456 = day_path+'/456Fitparams'+day_path[day_path.rfind('/')+1:]+'.tsv'
                if not os.path.exists(fits456):
                    file = open(fits456,'w')
                    file.write('Date\tAlpha\tP0\th1\tmv\tT\tgamma\toffset\n')
                    file.close()
                file = open(fits456,'r')
                file.readline()
                for line in file:
                    line = line.strip().split('\t')
                    lines[line[0]]=line
                file.close()
                lines[date] = data
                order = list(lines.keys())
                order.sort()
                file = open(fits456,"w")
                file.write('Date\tAlpha\tP0\th1\tmv\tT\tgamma\toffset\n')
                for j in range(len(order)-1):
                    for i in range(len(lines[order[j]])-1):
                        file.write(lines[order[j]][i])
                        file.write('\t')
                    file.write(lines[order[j]][-1])
                    file.write('\n')
                for i in range(len(lines[order[-1]])-1):
                    file.write(lines[order[-1]][i])
                    file.write('\t')
                file.write(lines[order[-1]][-1])
                file.close()
                print('456 param saved')
            else:
                lines = {}
                date = self.par_folder[self.par_folder.rfind('/')+1:]
                temp = list(map(str, self.fitted_param))
                data = [date]
                data.extend(temp)
                day_path = self.par_folder[:self.par_folder.rfind('/')]
                fits894 = day_path+'/894Fitparams'+day_path[day_path.rfind('/')+1:]+'.tsv'
                if not os.path.exists(fits894):
                    file = open(fits894,'w')
                    file.write('Date\tAlpha\tP0\th1\tmv\tT\tgamma\toffset\n')
                    file.close()
                file = open(fits894,'r')
                file.readline()
                for line in file:
                    line = line.strip().split('\t')
                    lines[line[0]]=line
                file.close()
                lines[date] = data
                order = list(lines.keys())
                order.sort()
                file = open(fits894,"w")
                file.write('Date\tAlpha\tP0\th1\tmv\tT\tgamma\toffset\n')
                for j in range(len(order)-1):
                    for i in range(len(lines[order[j]])-1):
                        file.write(lines[order[j]][i])
                        file.write('\t')
                    file.write(lines[order[j]][-1])
                    file.write('\n')
                for i in range(len(lines[order[-1]])-1):
                    file.write(lines[order[-1]][i])
                    file.write('\t')
                file.write(lines[order[-1]][-1])
                file.close()
                print('894 param saved')

            
