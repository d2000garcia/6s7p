import numpy as np
from numpy.polynomial import Polynomial as poly
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from scipy.special import voigt_profile as voigt
from scipy.special import wofz as wofz
from scipy.integrate import quad
import scipy as sci
from scipy import optimize as opt
from matplotlib import lines as lines
from numpy import pi as pi
import os as os

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
    return poly.fit(new_indices,new_data,1)

def cvt_abs_wav_to_diff(abs_wav):
    #assume units of wavnum are 1/cm
    c=299792458
    x = [0]
    for i in range(len(abs_wav)-1):
        x.append((abs_wav[i+1]-abs_wav[i])*c/10000000)
    return x #returns GHz freq diff


class data:
    def __init__(self, par_folder,BeatRunAvgN=100, beatnote_det_f=0, beat_rng=[0,8000], back_rngs=[[0,0],[6300,8000]], file_skip_lines=0, scan='456', F =0, exists = False):
        self.fitted = False
        
        if not exists:
            if scan == '456':
                filename = par_folder + r"\456Scan.csv"
            else:
                filename = par_folder + r"\894Scan.csv"

            if beatnote_det_f == 0:
                if scan == '456':
                    beatnote_det_f = 0.045
                else:
                    beatnote_det_f = 0.020
            self.par_folder = par_folder
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
            self.beat_height = 0
            if F == 0:
                if scan == '456':
                    self.set_transition(F1=4)
                else:
                    self.set_transition(F1=3)

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
                    beatnote_det_f = 0.045
                else:
                    beatnote_det_f = 0.020
            if F == 0:
                if scan == '456':
                    self.set_transition(F1=4)
                else:
                    self.set_transition(F1=3)
            self.beatnote_det_f = beatnote_det_f
            self.folder = folder
            self.indices = np.loadtxt(folder+r'\indices.csv', dtype=int, delimiter=',')
            self.scaledT = np.loadtxt(folder+r'\fitting\processed\scaledT.csv', delimiter=',')
            self.filteredBeat = np.loadtxt(folder+r'\beatnote\processed\filteredBeat.csv', delimiter=',')
            self.peak_indices2 = np.loadtxt(folder+r'\beatnote\processed\peak_indices.csv', dtype=int, delimiter=',').tolist()
            self.peak_val2 = np.loadtxt(folder+r'\beatnote\processed\peak_val.csv', delimiter=',').tolist()
            self.BeatRunAvgN = BeatRunAvgN
            self.par_folder = par_folder
            self.beat_height = 0

            if os.path.exists(self.folder+r'\fitting\processed\fitting_param.csv'):
                self.fitted=True
            else:
                self.fitted = False

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
            if self.fitted:
                self.fitted_param = np.loadtxt(self.folder+r'\fitting\processed\fitting_param.csv', delimiter=',')
                self.pcov = np.loadtxt(self.folder+r'\fitting\processed\pcov.csv',delimiter=',')
                self.alph_err=np.sqrt(np.diag(self.pcov))[1]
                self.alpha=self.fitted_param[1]
            # self.voigt_range = [max(self.back_rngs[0][1],self.beat_rng[0]), min(self.back_rngs[1][0],self.beat_rng[1])]
        
        if not os.path.exists(folder+r'\plots\VapPres.png'):
            temps = simple_dat_get(par_folder +r'\Temperature.csv',0)
            temp_0 = (20*temps[:,2]).tolist()
            temp_1 = (20*temps[:,3]).tolist()
            temp_2 = (20*temps[:,4]).tolist()
            ones = np.ones(len(temp_0))
            plt.scatter(ones,temp_0,color='k')
            plt.scatter(ones+1, temp_1,color='k')
            plt.scatter(ones+2, temp_2,color='k')
            plt.scatter([1,2,3], [np.mean(temp_0),np.mean(temp_1),np.mean(temp_2)],color='r')
            plt.title("Temperature Measurements")
            plt.savefig(folder+r'\plots\Temperature.png')
            plt.clf()

            plt.scatter(ones, list(map(vapor_pres,temp_0)),color='k')
            plt.scatter(ones+1, list(map(vapor_pres,temp_1)),color='k')
            plt.scatter(ones+2, list(map(vapor_pres,temp_2)),color='k')
            plt.scatter([1,2,3], list(map(vapor_pres,[np.mean(temp_0),np.mean(temp_1),np.mean(temp_2)])),color='r')
            plt.title("Vapor Pressure")
            plt.savefig(folder+r'\plots\VapPres.png')
            plt.clf()
            
            
    
    def reprocess_beatnote(self):
        if self.beat_height == 0:
            standard_peak_min = np.std(self.filteredBeat[self.BeatRunAvgN+1:len(self.indices)-self.BeatRunAvgN-1])*2
        else:
            standard_peak_min = self.beat_height
        peak_indices2 = find_peaks(self.filteredBeat, height=standard_peak_min,distance=20)
        peak_val2 = peak_indices2[1]['peak_heights']
        peak_indices2 = peak_indices2[0]
        # (peak_indices2, peak_val2) = correct_peaks(peak_indices=peak_indices2.tolist(), peak_val=peak_val2.tolist())
        self.peak_indices2 = peak_indices2
        self.peak_val2 = peak_val2

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
        plt.title(r'$\frac{Transmission}{Laser Power}$')
        plt.axvspan(0,self.beat_rng[0],alpha=0.2,color='grey')
        plt.axvspan(self.beat_rng[1],len(self.indices),alpha=0.2,color='grey')
        plt.savefig(self.folder+r'\plots\scaledT.png')
        plt.clf()
        if self.par_folder.find('BaselineMeas')!=-1:
            #is baseline measurement
            self.PwrWings = 0
        else:
            #not baseline measurement
            if self.par_folder+r'\PwrWings'+self.scan+'.csv' in os.listdir(self.par_folder):
                temp = np.loadtxt(self.par_folder+r'\PwrWings'+self.scan+'.csv', delimiter=',').tolist()
                self.PwrWings = np.mean(temp)
            else:
                self.PwrWings = 0

        # self.voigt_rng = [max(self.back_rngs[0][1],self.beat_rng[0]), min(self.back_rngs[1][0],self.beat_rng[1])]
            # plt.show()
        if self.par_folder.find('BaselineMeas')!=-1:
            #If this is the baseline measurement folder
            temp  = find_peaks(-np.array(self.scaledT), width=500)
            print(self.scan+ ' Pwr in wings mean: ' +str(np.mean(self.scaledT[temp[0][0]-250:temp[0][0]+250])))
            print(poly.fit(self.indices[temp[0][0]-250:temp[0][0]+250],self.scaledT[temp[0][0]-250:temp[0][0]+250],1))
            # np.savetxt(self.par_folder+r'\PwrWings'+self.scan+'.csv',self.scaledT[temp[0][0]-250:temp[0][0]+250])
            for i in os.listdir(self.par_folder[:self.par_folder.rfind('/')]):
                np.savetxt( self.par_folder[:self.par_folder.rfind('/')] +'/'+ i + '/PwrWings'+self.scan+'.csv', self.scaledT[temp[0][0]-250:temp[0][0]+250], delimiter=',')
            
            
    
    def calculate_beat_fit(self):
        if not type(self.peak_indices2) == list:
            (self.cleared_indices, self.cleared_peaks) = cutoff_ends(self.peak_indices2.copy().tolist(), self.peak_val2.copy().tolist(), self.beat_rng[0], self.beat_rng[1])
        else:(self.cleared_indices, self.cleared_peaks) = cutoff_ends(self.peak_indices2.copy(), self.peak_val2.copy(), self.beat_rng[0], self.beat_rng[1])
        # (self.cleared_indices, self.cleared_peaks)=(self.peak_indices2.copy(), self.peak_val2.copy())
        self.differences = list(map(lambda x1,x2:x1-x2, self.cleared_indices[1:], self.cleared_indices[:len(self.cleared_indices)-1]))
        avg_diff = np.mean(self.differences)
        self.freq = [0]
        for diff in self.differences:
            if diff < avg_diff:
                self.freq.append(self.freq[-1] + self.beatnote_det_f*2)
            else:
                self.freq.append(self.freq[-1] + 0.250 - self.beatnote_det_f*2)
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

        # self.voigt_rng = [max(self.back_rngs[0][1],self.beat_rng[0]), min(self.back_rngs[1][0],self.beat_rng[1])]

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

    def set_transition(self, F1):
        #Can be used to verify i some fashion if need be of the coefficient
        # fine_structure = 0.00729735256
        # Lcell = 29.09 #cm
        # denom = (7+1) * (1+2) #(2I+1)(2J+1) #I=7/2, J=1/2 6s starting ground state

        #Voigt(x,s,g)= Re(w(z))/s \rad(2pi) , z = (x + i g)/(s\rad(2)) ; w = Faddeeva function
        if self.scan == '456':
            #In GHz
            if F1 == 3:
                self.abs_wavenum = [21946.56347,21946.56514,21946.56736]
                #from W. Williams, M. Herd, and W. Hawkins, Laser Phys. Lett. 15,095702 (2018).
                self.center = np.mean(self.abs_wavenum) * 29.9792458
                self.hypsplit = cvt_abs_wav_to_diff(self.abs_wavenum)
                self.hyp_weights = [5/12,7/26,5/16]
                #Higher frequency because its closer ie. F=3->F=2,3,4
            else:
                #F1 == 4
                self.abs_wavenum = [21946.25850,21946.26072,21946.26349]
                self.center = np.mean(self.abs_wavenum) * 29.9792458
                self.hypsplit = cvt_abs_wav_to_diff(self.abs_wavenum)
                self.hyp_weights = [7/48,7/16,5/16]
                #Lower frequency because father F=4->F=3,4,5
        elif self.scan == '894':
            main_tran = 335116.048807
            center_to6PF3 = 0.656820
            center_to6PF4 = 0.510860
            center_to6SF3 = 5.170855370625
            center_to6SF4 = 4.021776399375
            if F1 == 3:
                self.abs_wavenum = [main_tran+center_to6SF3-center_to6PF3,main_tran+center_to6SF3+center_to6PF4]
                self.center = np.mean(self.abs_wavenum)
                self.hyp_weights = [7/24,7/8]
                self.hypsplit = [0,1.167680]
                #Higher frequency because its closer ie. F=3->F=3,4
            else:
                #F1 == 4
                self.abs_wavenum = [main_tran-center_to6SF4-center_to6PF3,main_tran-center_to6SF4+center_to6PF4]
                self.center = np.mean(self.abs_wavenum)
                self.hyp_weights = [7/8,5/8]
                # self.hyp_weights = [5/8,7/8]
                self.hypsplit = [0,1.167680]
                #Lower frequency because father F=4->F=3,4 

        
    
    def set_fitting_function(self):
        #constants wihout powers
        c=2.99792458
        afs=7.29735256
        m=2.20694650
        kB=1.3806503
        k1 = np.sqrt(kB/m/c**2) * 10**(-7) #to be used for delta _wD = w *k1 *sqrt(T)
        k2 = 10000 * afs * np.sqrt(m*c*c*pi**3/(8*kB)) #power analysis leads to the 10^4 factor 
        
        temp = 273+30  # guess at hot portion of cell
        if self.scan == '894':
            mini = find_peaks(-np.array(self.scaledT), width=150,height=np.mean(-self.scaledT))
            guess = self.beatfit(mini[0][0])
        else:
            temp = 100
            mini = 0
            for i,val in enumerate(self.scaledT):
                if val < temp:
                    temp = val
                    mini = i
            guess = self.beatfit(mini)
            
        if self.scan == '456':
            w1 = self.abs_wavenum[0]*29.9792458 *k1 #Abs freq w1 in GHz
            p0 = 0.9 #scaledT pwr at top
            Life = 1/137.54/2 #half of lifetime in MHz
            wD0 = self.center * np.sqrt(380) * k1 #estimate
            denom = self.center* k1
        else:
            w1 = self.abs_wavenum[0] * k1
            p0 = 0.11 #scaledT power at top
            Life = 0.0045612/2 #half of lifetime in MHz
            wD0 = self.center * np.sqrt(380) * k1
            
        coeff = self.hyp_weights
        param_guess = [1,0.4,1/10,0.7]
        sqrtlog2 = np.sqrt(np.log(2))
        temp = np.vectorize( lambda x,b:complex(x,b), excluded={'b'}, cache=True)
        # self.voit_eqn = 
                #x1 = coeff
                #x2 = freq shifts from peak 1
                #-mv shifts to right considering out scan starts below peak 1 freq
                #+shift moves entire profile peak 1 freq to left so that the actual fitting is around 0 freq
                #so w + shift - mv ideally moves peaks to relative frequency from start of scan
                #wD is the temperature dependentr part of the Dopler full width half max, since it has frequency portion include small shiftings
        # self.fitting_eqn = lambda w,p0,a,wD,L,mv: p0 * np.exp(-a * (w+shift -mv) * np.sum(np.real(np.array(list(map(lambda x1,x2:x1*wofz(sqrtlog2*complex(2*(w-x2-mv), L)/(w+shift - mv)/wD)/(w+shift - mv)/wD,coeff,self.hypsplit))))))

        self.fitting_eqn3 = lambda w,p0,a,wD,mv,mv2,k0, offset: p0 * (1-k0*(w - mv - mv2)) * np.exp(-a * np.sum(np.array(list(map(lambda x1,x2:x1*wofz(temp(w-x2-mv, Life)/(np.sqrt(2)*wD))/(np.sqrt(2*pi)*wD),self.hyp_weights,self.hypsplit))),axis=0).real) + offset

        # self.fitting_eqn4 = lambda w,p0,a,wD,mv,mv2,k0, offset: p0 * (1-k0*(w - mv - mv2)) * np.exp(-a * np.sum(np.array(list(map(lambda x1,x2,x3:x1*wofz(temp(w-x2-mv, Life)/(np.sqrt(2)*wD))/(np.sqrt(2*pi)*wD),self.hyp_weights,self.hypsplit))),axis=0).real) + offset

        param_guess3 = [p0,0.1,wD0,guess,0.01,0.01,0.1]
        if self.scan == '456':
            bounds3 = ([0.1,0.00001,0.1,0.00001,-2,-1,0],[3,10,0.5,10,2,1,0.2])
        else:
            bounds3 = ([0.01,0.00001,0.05,0.00001,-2,-1,0],[3,10,0.3,4,2,1,0.2])

        plotting_freq = self.beatfit(np.array(self.indices[self.beat_rng[0]:self.beat_rng[1]]))

        fitted_param3, pcov3 = opt.curve_fit(self.fitting_eqn3, plotting_freq,self.scaledT[self.beat_rng[0]:self.beat_rng[1]],param_guess3,bounds=bounds3)
        print('fitted params')
        print(fitted_param3)
        self.fitted_param = fitted_param3
        # print(fitted_param3)
        self.pcov = pcov3
        self.alpha = fitted_param3[1]
        self.fitted = True
        perr = np.sqrt(np.diag(pcov3))
        # print(perr)
        # self.alph_err=perr[1]
        print(np.linalg.cond(pcov3))
        
        resid = self.fitting_eqn3(plotting_freq,*fitted_param3)-self.scaledT[self.beat_rng[0]:self.beat_rng[1]]
        chi2 = resid**2
        self.alph_err=np.sum(chi2)/(self.beat_rng[1]-self.beat_rng[0])
        np.savetxt(self.folder+r'\fitting\processed\fitting_param.csv', self.fitted_param, delimiter=',')
        np.savetxt(self.folder+r'\fitting\processed\pcov.csv',self.pcov,delimiter=',')
        plt.scatter(plotting_freq,self.scaledT[self.beat_rng[0]:self.beat_rng[1]])
        plt.plot(plotting_freq,self.fitting_eqn3(plotting_freq,*fitted_param3), '-r',linewidth=0.5,marker='.')#, mew='0.05')
        plt.title(self.scan+ 'Fitted plot, a='+str(fitted_param3[1]) +r', $\chi^2=$'+ str(self.alph_err))
        plt.xlabel('Freq [GHz]')
        # plt.show()
        plt.savefig(self.folder+r'\plots\FittedScan.png')
        plt.clf()

        plt.scatter(plotting_freq,resid)
        # plt.show()
        plt.title(self.scan+ 'Fitted plot residuals')
        plt.xlabel('Freq [GHz]')
        plt.savefig(self.folder+r'\plots\FittedScanResid.png')
        plt.clf()

        if self.scan == '456':
            lines = {}
            date = self.par_folder[self.par_folder.rfind('/')+1:]
            temp = list(map(str, fitted_param3.copy().tolist()))
            data = [date]
            data.extend(temp)
            day_path = self.par_folder[:self.par_folder.rfind('/')]
            fits456 = day_path+'/456Fitparams'+day_path[day_path.rfind('/')+1:]+'.tsv'
            if not os.path.exists(fits456):
                file = open(fits456,'w')
                file.write('Date\tAlpha\twD\tmv\tmv2\tk0\toffset\n')
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
            file.write('Date\tAlpha\twD\tmv\tmv2\tk0\toffset\n')
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

        
