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
    peak_indices2 = find_peaks(filteredBeat, height=standard_peak_min,distance=20)
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
    def __init__(self, par_folder,BeatRunAvgN=100, beatnote_det_f=0, beat_rng=[0,8000], back_rngs=[[0,0],[6300,8000]], file_skip_lines=0, scan='456', F1=0, exists = False,T=[]):
        