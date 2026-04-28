import os as os
import numpy as np
from numpy.polynomial import Polynomial as poly
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
from numpy import pi as pi
from PIL import Image, ImageTk
from scipy.signal import find_peaks
from scipy.special import wofz as wofz
from scipy.integrate import quad
from scipy import optimize as opt
from matplotlib import lines as lines
from numpy import pi as pi
import os as os
import lmfit as lm 

Day_folder = 'test'
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


def numerisize_date(contents):
    #Input list for dates
    temp = []
    for meas in contents:
        time =0
        if 'PM' or 'AM' in meas:
            dat = meas.split('+')[1].split('-')
            if 'PM' in meas:
                time = 12
                time+= float(dat[2].split('PM')[0])/3600
            else:
                time+= float(dat[2].split('AM')[0])/3600
            time += float(dat[0])%12
            time += float(dat[1])/60
        else:
            time = float(meas.split('+')[1])
        temp.append(time)
    return temp

# def BackgroundFit(data_bounds, indices, data, scan):
#     #b = self.indices[-1]
#     #t = 2(x-a)/(b-a) -1 scales x in [a,b] to [-1,1]
#     #Fit becomes y = alpha1 t + alpha2
#     #alpha1 = k1 *a
#     #alpha2 = k1 *(b+a)/2 + k2
#     #truefit  y = k1 x + k2
#     # scaled_indices = 2*self.indices/b -1

#     ###         Note numpy.polynomial.Polynomial.fit offers built in scaling and shifting of data
#     ###                                 more numerically stable
#     new_data = []
#     new_indices = []
#     for bound in data_bounds:
#         new_data.extend(data[bound[0]:bound[1]])
#         new_indices.extend(indices[bound[0]:bound[1]])
#     upper = np.mean(data[data_bounds[1][0]:data_bounds[1][1]])-np.mean(data[data_bounds[0][0]:data_bounds[0][1]])
#     lower = np.mean(indices[data_bounds[1][0]:data_bounds[1][1]])-np.mean(indices[data_bounds[0][0]:data_bounds[0][1]])
#     if scan == '456':
#         fitted_param, pcov = opt.curve_fit(lambda x,k2,k,b:k2*x**2 + k*x+b, new_indices,new_data,[0.01,upper/lower,new_data[0]])
#     else:
#         fitted_param, pcov = opt.curve_fit(lambda x,k,b:k*x+b, new_indices,new_data,[upper/lower,new_data[0]])
#     return fitted_param

class plots:
    def __init__(self,window,default_path = r".\Picture_template.png", plot_w = 500, plot_h = 300):
        default_img = Image.open(default_path)
        resized_default = default_img.resize((plot_w, plot_h), Image.LANCZOS)
        #Save variables for reference
        self.window = window
        self.plot_w = plot_w
        self.plot_h = plot_h
        #labels to ref plot indices
        self.plotslabs=['TAvg','TAll','Time']
        self.type=['Cold','Hot']
        self.day_fold = ''
        self.window_manager={}
        i=-1
        for type in self.type:
            i+=1
            self.window_manager[type]={}
            self.window_manager[type]['Note']=ttk.Notebook(window)
            self.window_manager[type]['Note'].grid(column=i,row=0,columnspan=1, sticky="nsew")
            self.window_manager[type]['Imgs'] = {}
            for name in self.plotslabs:
                #getting default image to load
                self.window_manager[type]['Imgs'][name] = {}
                self.window_manager[type]['Imgs'][name]['TkImg']=ImageTk.PhotoImage(resized_default.copy())
                self.window_manager[type]['Imgs'][name]['Label']= tk.Label(self.window_manager[type]['Note'],image=self.window_manager[type]['Imgs'][name]['TkImg'])
                self.window_manager[type]['Imgs'][name]['Label'].image = self.window_manager[type]['Imgs'][name]['TkImg']
                self.window_manager[type]['Imgs'][name]['Label'].pack()
                self.window_manager[type]['Note'].add(self.window_manager[type]['Imgs'][name]['Label'],text=type+' '+name)

        # self.window_manager = {'Cold':{'Note':ttk.Notebook(window)},'Hot':{'Note':ttk.Notebook(window)}}
        # self.window_manager['Cold']['Note'].grid(column=0,row=0,columnspan=1, sticky="nsew")
        # self.window_manager['Hot']['Note'].grid(column=1,row=0,columnspan=1, sticky="nsew")
        # self.window_manager['Cold']['']    
    
    def update_working_dir(self, new_day_fold):
        #to be called when picking new day data set
        self.day_fold = new_day_fold
        self.fold = self.day_fold + '\\TemperaturePlots'
    
    def change_Label_image(self,new,oldlabel):
    #oldlabel is the label you want to change and
    #new is new Tkimage to exchange
        oldlabel.configure(image=new)
        oldlabel.image = new
    
    def update_all_imgs(self):
        for type in self.type:
            for name in self.plotslabs:
                if name != 'TBD':
                    plot_path = self.fold + '\\' + type +name + '.png'
                    temp = Image.open(plot_path)
                    resized_temp = temp.resize((self.plot_w, self.plot_h), Image.LANCZOS)
                    self.window_manager[type]['Imgs'][name]['TkImg'] = ImageTk.PhotoImage(resized_temp)
                    self.change_Label_image(self.window_manager[type]['Imgs'][name]['TkImg'],self.window_manager[type]['Imgs'][name]['Label'])


    # def update_image(self, tochange):
    #     #way to update images after picking new folder or generating new pictures after they're saved
    #     for whichone in tochange:
    #         plot_path = self.fold + '\\' + whichone + '.png'
    #         temp = Image.open(plot_path)
    #         resized_temp = temp.resize((self.plot_w, self.plot_h), Image.LANCZOS)
    #         if whichone == 'TColdAvg':
    #             self.plots[0] = ImageTk.PhotoImage(resized_temp)
    #         elif whichone == 'THotAvg':
    #             self.plots[1] = ImageTk.PhotoImage(resized_temp)
    #         # elif whichone == 'scaledT':
    #         #     self.plots[2] = ImageTk.PhotoImage(resized_temp)
    #         # elif whichone == 'FittedScan':
    #         #     self.plots[3] = ImageTk.PhotoImage(resized_temp)


class ResidAnalysis:
    def __init__(self,window,img_scale=1):
        self.day_folder = ''
        self.folderpath = ''
        self.folderpath_tkvar = tk.StringVar()
        self.window = window
        self.plots = plots(window=window,plot_w=int(500*img_scale),plot_h=int(300*img_scale))
        self.scans = ['456','894']
        self.fitted_params = [0,0]
        self.F1 = [0,0]
        self.scaledT = [0,0]
        self.resid = [0,0]
        self.beatfit = [0,0]
        self.indices = [0,0]
        self.scan_folder = [0,0]
        self.fit_rng = [0,0]
        self.plotting_freq = [0,0]
        self.functions = [0,0]
        self.peaks = [0,0]
        self.properties = [0,0]
        self.peak_fwhm = [0,0]
        self.resid_stat = [[0,0,0],[0,0,0]]

    def print_path(self):
        print(self.folderpath)

    def checkforanalysis(self):
        months = ['Apr','Mar','Sep','Oct','Nov','Aug']
        found = -1
        if not self.day_folder == '':
            if os.path.exists(self.day_folder):
                contents = os.listdir(path = self.day_folder)
                self.data = []
                for content in contents:
                    self.folderpath = self.day_folder+ '\\' +content
                    if os.path.exists(self.folderpath+r'\Analysis\456\fitting\processed\fitting_param.csv') and os.path.exists(self.folderpath+r'\Analysis\894\fitting\processed\fitting_param.csv'):
                        #its been fit previously
                        if (not os.path.exists(self.folderpath+r'\Analysis\456\fitting\processed\Residuals.csv')) or (not os.path.exists(self.folderpath+r'\Analysis\894\fitting\processed\Residuals.csv')):
                            self.read_in_data()
                            self.set_transition()
                            self.calculate_residuals()
                        self.get_peaks()
                        self.RMSE()
                self.save_stats()
                print('Done Saving')
                        

    def set_transition(self):
        #Can be used to verify i some fashion if need be of the coefficient
        # fine_structure = 0.00729735256
        # Lcell = 29.09 #cm
        # denom = (7+1) * (1+2) #(2I+1)(2J+1) #I=7/2, J=1/2 6s starting ground state

        #Voigt(x,s,g)= Re(w(z))/s \rad(2pi) , z = (x + i g)/(s\rad(2)) ; w = Faddeeva function
        for i in [0,1]:
            #New function using lmfit
            #constants wihout powers
            c=2.99792458
            afs=7.29735256
            m=2.2069484567911638
            kB=1.3806503
            k1 = np.sqrt(kB/m/c**2) * 10**(-7) #to be used for delta _wD = w *k1 *sqrt(T)
            if self.scans[i] == '456':
                #In GHz
                if self.F1[i] == 3:
                    abs_wavenum = [21946.56347,21946.56514,21946.56736]
                    #from W. Williams, M. Herd, and W. Hawkins, Laser Phys. Lett. 15,095702 (2018).
                    abs_freq = list(map(lambda x:  x*29.9792458,abs_wavenum))#29.9..... = c decimals for scaling to convert to GHz

                    hypsplit = list(map(lambda x: x-abs_freq[0],abs_freq)) 
                    hyp_weights = [5/12,7/26,5/16]
                    #Higher frequency because its closer ie. F=3->F=2,3,4
                else:
                    #F1 == 4
                    abs_wavenum = [21946.25850,21946.26072,21946.26349]
                    abs_freq = list(map(lambda x:  x*29.9792458,abs_wavenum))#29.9..... = c decimals for scaling for scaling to convert to GHz

                    hypsplit = list(map(lambda x: x-abs_freq[0],abs_freq)) 
                    hyp_weights = [7/48,7/16,5/16]
                    #Lower frequency because father F=4->F=3,4,5
                if len(self.fitted_params) == 8:
                    self.functions[i] = lambda w,a,p0,h1,h2,mv,T,gamma,base: p0*(1+h1*w+h2*w**2)*np.exp(-a*((w-mv+abs_freq[0])/10**6)*(voigt(w,hyp_weights[0],mv,np.sqrt(T+273.15)*k1*abs_freq[0],gamma)+
                                                                            voigt(w,hyp_weights[1],mv+hypsplit[1],np.sqrt(T+273.15)*k1*abs_freq[1],gamma)+
                                                                            voigt(w,hyp_weights[2],mv+hypsplit[2],np.sqrt(T+273.15)*k1*abs_freq[2],gamma))) + base
                else:
                    self.functions[i] = lambda w,a,p0,h1,mv,T,gamma,base: (p0+h1*w)*np.exp(-a*((w-mv+abs_freq[0])/10**6)*(voigt(w,hyp_weights[0],mv,np.sqrt(T+273.15)*k1*abs_freq[0],gamma)+
                                                                            voigt(w,hyp_weights[1],mv+hypsplit[1],np.sqrt(T+273.15)*k1*abs_freq[1],gamma)+
                                                                            voigt(w,hyp_weights[2],mv+hypsplit[2],np.sqrt(T+273.15)*k1*abs_freq[2],gamma))) + base
            elif self.scans[i] == '894':
                if self.F1[i] == 0:
                    peaks, properties = find_peaks(-self.scaledT[i],width=500, prominence=0.1)
                    if (properties['right_ips'][1] - properties['left_ips'][1]) > (properties['right_ips'][0] - properties['left_ips'][0]):
                        #peak 2 is larger 
                        self.F1[i]=3
                    else:
                        #peak 1 is larger
                        self.F1[i] = 4
                #already in GHz
                main_tran = 335116.048807
                center_to6PF3 = 0.656820
                center_to6PF4 = 0.510860
                center_to6SF3 = 5.170855370625
                center_to6SF4 = 4.021776399375
                if self.F1[i] == 3:
                    abs_freq2= [main_tran+center_to6SF3-center_to6PF3,main_tran+center_to6SF3+center_to6PF4]
                    hyp_weights2 = [7/24,7/8]
                    hypsplit2 = [0,1.167680]
                    #Higher frequency because its closer ie. F=3->F=3,4
                else:
                    #F1 == 4
                    abs_freq2 = [main_tran-center_to6SF4-center_to6PF3,main_tran-center_to6SF4+center_to6PF4]
                    hyp_weights2 = [7/8,5/8]
                    hypsplit2 = [0,1.167680]
                    #Lower frequency because father F=4->F=3,4 
                self.functions[i] = lambda w,a,p0,h1,mv,T,gamma,base: (p0+h1*w)*np.exp(-a*((w-mv+abs_freq2[0])/10**6)*(voigt(w,hyp_weights2[0],mv,np.sqrt(T+273.15)*k1*abs_freq2[0],gamma)+
                                                                                                            voigt(w,hyp_weights2[1],mv+hypsplit2[1],np.sqrt(T+273.15)*k1*abs_freq2[1],gamma))) + base
    def read_in_data(self):
        for i in [0,1]:
            self.scan_folder[i] = self.folderpath + '\\Analysis\\' + self.scans[i]
            self.scaledT[i] = np.loadtxt(self.scan_folder[i]+r'\fitting\processed\scaledT.csv', delimiter=',')
            self.indices[i] = np.loadtxt(self.scan_folder[i]+r'\indices.csv', dtype=int, delimiter=',')
            temp = np.loadtxt(self.scan_folder[i]+r'\beatnote\processed\beat_fit_param.csv', delimiter=',')
            self.beatfit[i] = poly(temp[4:], temp[0:2], temp[2:4])
            self.fitted_params[i] = np.loadtxt(self.scan_folder[i]+r'\fitting\processed\fitting_param.csv', delimiter=',').tolist()
            self.fit_rng[i] = np.loadtxt(self.scan_folder[i]+r'\entries\beat_rng.csv', dtype=int, delimiter=',').tolist()
            self.plotting_freq[i] = self.beatfit[i](np.array(self.indices[i][self.fit_rng[i][0]:self.fit_rng[i][1]]))
            # print(self.fitted_params[i])
            # if i == 1:
                # np.savetxt(r'C:\Users\Wolfwalker\Documents\git\6s7p\temp1.csv',self.plotting_freq[1])
    
    def calculate_residuals(self):
        for i in [0,1]:
            fit = self.functions[i](self.plotting_freq[i],*self.fitted_params[i])
            np.savetxt(self.scan_folder[i]+r'\fitting\processed\Fit.csv', fit, delimiter=',')
            np.savetxt(self.scan_folder[i]+r'\fitting\processed\plotting_freq.csv', self.plotting_freq[i], delimiter=',')
            self.resid[i] = fit - self.scaledT[i][self.fit_rng[i][0]:self.fit_rng[i][1]]
            # plt.scatter(self.plotting_freq[i],self.scaledT[i][self.fit_rng[i][0]:self.fit_rng[i][1]])
            # plt.plot(self.plotting_freq[i],self.functions[i](self.plotting_freq[i],*self.fitted_params[i]),'-r')
            # plt.show()
            # plt.plot(self.plotting_freq[i],self.resid[i],'-b')
            # plt.show()
            # plt.plot(self.plotting_freq[i],self.resid[i],'-b')
            np.savetxt(self.scan_folder[i]+r'\fitting\processed\Residuals.csv', self.resid[i], delimiter=',')

    def get_peaks(self):
        for i in [0,1]:
            self.scan_folder[i] = self.folderpath + '\\Analysis\\' + self.scans[i]
            self.scaledT[i] = np.loadtxt(self.scan_folder[i]+r'\fitting\processed\scaledT.csv', delimiter=',').tolist()
            self.fit_rng[i] = np.loadtxt(self.scan_folder[i]+r'\entries\beat_rng.csv', dtype=int, delimiter=',').tolist()
            self.peaks[i], self.properties[i] = find_peaks(-np.array(self.scaledT[i][self.fit_rng[i][0]:self.fit_rng[i][1]]),width=500,prominence=0.02)
            if i == 0:
                self.peak_fwhm[i] = int(self.properties[i]['right_ips'][0]-self.properties[i]['left_ips'][0])
            elif i == 1:
                self.peak_fwhm[i] = int(self.properties[i]['right_ips'][1]-self.properties[i]['left_ips'][0])
            
    def RMSE(self):
        #Root mean squared error
        for i in [0,1]:
            self.resid[i] = np.loadtxt(self.scan_folder[i]+r'\fitting\processed\Residuals.csv', delimiter=',')
            temp = self.resid[i]**2
            if i == 0:
                around_peak = [self.peaks[i][0]-int(self.peak_fwhm[i]*1.5),self.peaks[i][0]+int(self.peak_fwhm[i]*1.5)]
                self.resid_stat[i][0] = np.sqrt(np.sum(temp)/(temp.size-8))
                temp2 = temp[around_peak[0]:around_peak[1]]
                self.resid_stat[i][1] = np.sqrt(np.sum(temp2)/(temp2.size-8))
                self.resid_stat[i][2] = max(abs(self.resid[i]))
            else:
                around_peak = [int((self.peaks[i][0]+self.peaks[i][1])/2-self.peak_fwhm[i]*1.5),int((self.peaks[i][0]+self.peaks[i][1])/2+self.peak_fwhm[i]*1.5)]
                self.resid_stat[i][0] = np.sqrt(np.sum(temp)/(temp.size-7))
                temp2 = temp[around_peak[0]:around_peak[1]]
                self.resid_stat[i][1] = np.sqrt(np.sum(temp2)/(temp2.size-7))
                self.resid_stat[i][2] = max(abs(self.resid[i]))
        print(self.resid_stat)
        date = self.folderpath[self.folderpath.rfind('\\')+1:]
        self.data.append([date])
        self.data[-1].extend(list(map(str,self.resid_stat[0])))
        self.data[-1].extend(list(map(str,self.resid_stat[1])))
    
    def save_stats(self):
        file = open(self.day_folder +r'\fit_resid.tsv','w')
        file.write('date \t 456 RMSE \t 456 peak RMSE \t 456 max resid \t 894 RMSE \t 894 peak RMSE \t 894 max resid \n')
        for line in self.data:
            for thing in line:
                file.write(thing + '\t')
            file.write('\n')
        file.close()

        
    def open_file_dialog(self):
        temporary = filedialog.askdirectory(
            initialdir="/",  # Optional: set initial directory
            title="Select a folder",
            # filetypes=(("Text files", "*.txt"), ("All files", "*.*")) # Optional: filter file types
        )
        self.day_folder = temporary
        if self.day_folder!='':
            date = self.day_folder[self.day_folder.rfind('/')+1:]
            if os.path.exists(self.day_folder+r'\Fine.txt'):
                file = open(self.day_folder+r'\Fine.txt','r')
                Fine = list(map(int,file.readline().split(',')))
                file.close()
            else:
                Fine = [3,4]
            self.F1 = Fine
            self.checkforanalysis()

first = True
scale = 1.6
if __name__ == '__main__':
    root = tk.Tk()
    if first:
        first = False
        analysis = ResidAnalysis(window=root,img_scale=scale)
        root.title("Residual Calculations")

        open_button = ttk.Button(root, text="Data Folder", command= analysis.open_file_dialog)
        open_button.grid(column=0,row=1)
        
        open_button5 = ttk.Button(root, text="Close", command= exit)
        open_button5.grid(column=1,row=1)
        
        # workingdir_txt = tk.StringVar()
        # workingdir_txt.set('hello')
        # workingdir = ttk.Label(root, textvariable=workingdir_txt)
        # workingdir.grid(column=3, row=11,columnspan=3, sticky="nsew")
    root.mainloop()
	