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

    def print_path(self):
        print(self.folderpath)

    def checkforanalysis(self):
        months = ['Apr','Mar','Sep','Oct','Nov','Aug']
        found = -1
        if not self.folderpath == '':
            if os.path.exists(self.folderpath):
                if os.path.exists(self.folderpath+r'\Analysis\456\fitting\processed\fitting_param.csv') and os.path.exists(self.folderpath+r'\Analysis\456\fitting\processed\fitting_param.csv'):
                    self.read_in_data()
                    self.set_transition()
                    self.calculate_residuals()

            # contents = os.listdir(path = self.folderpath)
            # for i,mon in enumerate(months):
            #     found = self.folderpath.find(mon) + 1
            #     if found:
            #         self.date = self.folderpath[found-1:found+10]
            # if 'TemperaturePlots' in contents:
            #     #Then its been run for this day already
            #     self.plots.update_working_dir(self.folderpath)
            #     self.plots.update_all_imgs()

            # else:
            #     self.work_folder = self.folderpath+'\\TemperaturePlots'
            #     os.mkdir(self.work_folder)
            #     colors = ['black','dodgerblue','crimson']
            #     #Need to run temperatures analysis
            #     # try:
            #     MeanPlots = [[0,0],[0,0]]
            #     MeanPlots[0][0] = plt.figure()
            #     MeanPlots[0][1] = plt.figure()
            #     MeanPlots[1][0] = MeanPlots[0][0].add_axes([0.1,0.1,0.75,0.8])
            #     MeanPlots[1][1] = MeanPlots[0][1].add_axes([0.1,0.1,0.8,0.8])
            #     meanplotdat = [[0,0],[0,0]]

            #     MeanPlotsTimeScale = [[0,0],[0,0]]
            #     MeanPlotsTimeScale[0][0] = plt.figure()
            #     MeanPlotsTimeScale[0][1] = plt.figure()
            #     MeanPlotsTimeScale[1][0] = MeanPlotsTimeScale[0][0].add_axes([0.1,0.1,0.75,0.8])
            #     MeanPlotsTimeScale[1][1] = MeanPlotsTimeScale[0][1].add_axes([0.1,0.1,0.8,0.8])

            #     AllPtPlots = [[0,0],[0,0]]
            #     AllPtPlots[0][0] = plt.figure()
            #     AllPtPlots[0][1] = plt.figure()
            #     AllPtPlots[1][0] = AllPtPlots[0][0].add_axes([0.1,0.1,0.75,0.8])
            #     AllPtPlots[1][1] = AllPtPlots[0][1].add_axes([0.1,0.1,0.8,0.8])

            #     vap_axis = [0,0,0]
            #     vap_axis[0] = AllPtPlots[1][0].twinx()
            #     vap_axis[1] = MeanPlots[1][0].twinx()
            #     vap_axis[2] = MeanPlotsTimeScale[1][0].twinx()

            #     for i in range(2):
            #         MeanPlots[1][i].set_xlabel('Measurment number')
            #         MeanPlots[1][i].set_ylabel('Temperature [C]')
            #         AllPtPlots[1][i].set_xlabel('Measurment number')
            #         AllPtPlots[1][i].set_ylabel('Temperature [C]')
            #         MeanPlotsTimeScale[1][i].set_xlabel('Time of Day')
            #         MeanPlotsTimeScale[1][i].set_ylabel('Temperature [C]')
            #         vap_axis[i].set_ylabel(r'log($P_v$) [log(Torr)]')
            #     vap_axis[2].set_ylabel(r'log($P_v$) [log(Torr)]')
            #     allptminmax = [[1000,-1000],[1000,-1000]]
            #     tempminmac = [[0,0],[0,0]]
            #     # print(contents)
            #     temp = len(contents)
            #     for num in range(temp):
            #         if ('.tsv' in contents[temp-num-1]) or ('xlsx' in contents[temp-num-1]):
            #             contents.pop(temp-num-1)
            #     # print(contents)
            #     times = numerisize_date(contents)
            #     for j,run in enumerate(contents):
            #         temp=np.loadtxt(self.folderpath+'\\'+run+'\\TemperatureV2.csv',delimiter=',')
            #         tempminmac[0][0] = np.min(temp[:,:3])
            #         tempminmac[0][1] = np.max(temp[:,:3])
            #         tempminmac[1][0] = np.min(temp[:,3:])
            #         tempminmac[1][1] = np.max(temp[:,3:])
            #         for k in range(2):
            #             if tempminmac[k][0] < allptminmax[k][0]:
            #                 allptminmax[k][0] = tempminmac[k][0]
            #             if tempminmac[k][1] > allptminmax[k][1]:
            #                 allptminmax[k][1] = tempminmac[k][1]
            #         #Assume for temperatureV2 for now
            #         shape = temp.shape[0]
            #         xallPt = np.linspace(j,j+1/3,shape)
            #         for i in range(3):
            #             meanplotdat[0][0] = np.mean(temp[:,i])
            #             meanplotdat[0][1] = np.std(temp[:,i])/np.sqrt(shape)
            #             meanplotdat[1][0] = np.mean(temp[:,i+3])
            #             meanplotdat[1][1] = np.std(temp[:,i+3])/np.sqrt(shape)
            #             #vline
            #             MeanPlots[1][0].plot([j+(1+2*i)/6,j+(1+2*i)/6],[meanplotdat[0][0]-meanplotdat[0][1],meanplotdat[0][0]+meanplotdat[0][1]],color=colors[i])
            #             MeanPlots[1][1].plot([j+(1+2*i)/6,j+(1+2*i)/6],[meanplotdat[1][0]-meanplotdat[1][1],meanplotdat[1][0]+meanplotdat[1][1]],color=colors[i])
            #             #hline
            #             MeanPlots[1][0].plot([j+i/3,j+(i+1)/3],[meanplotdat[0][0],meanplotdat[0][0]],color=colors[i])
            #             MeanPlots[1][1].plot([j+i/3,j+(i+1)/3],[meanplotdat[1][0],meanplotdat[1][0]],color=colors[i])
                        
            #             #Time correlated plot
            #             #vline
            #             MeanPlotsTimeScale[1][0].plot([times[j]-4/60,times[j]-4/60],[meanplotdat[0][0]-meanplotdat[0][1],meanplotdat[0][0]+meanplotdat[0][1]],color=colors[i])
            #             MeanPlotsTimeScale[1][1].plot([times[j]-4/60,times[j]-4/60],[meanplotdat[1][0]-meanplotdat[1][1],meanplotdat[1][0]+meanplotdat[1][1]],color=colors[i])
            #             #hline
            #             MeanPlotsTimeScale[1][0].plot([times[j]-8/60,times[j]],[meanplotdat[0][0],meanplotdat[0][0]],color=colors[i])
            #             MeanPlotsTimeScale[1][1].plot([times[j]-8/60,times[j]],[meanplotdat[1][0],meanplotdat[1][0]],color=colors[i])

            #             AllPtPlots[1][0].plot(xallPt+i/3,temp[:,i],color=colors[i])
            #             AllPtPlots[1][1].plot(xallPt+i/3,temp[:,i+3],color=colors[i])
            #     AllPtPlots[1][0].set_xlim(0,j)
            #     AllPtPlots[1][0].set_ylim(allptminmax[0][0],allptminmax[0][1])
            #     vap_axis[0].set_ylim(vapor_pres(allptminmax[0][0]),vapor_pres(allptminmax[0][1]))
            #     AllPtPlots[1][0].set_title('ColdFinger All Points, '+self.date)
            #     AllPtPlots[1][1].set_xlim(0,j)
            #     AllPtPlots[1][1].set_ylim(allptminmax[1][0],allptminmax[1][1])
            #     AllPtPlots[1][1].set_title('MainCell All Points, '+self.date)
            #     AllPtPlots[0][0].savefig(self.work_folder+'\\ColdTAll.png')
            #     AllPtPlots[0][1].savefig(self.work_folder+'\\HotTAll.png')
            #     AllPtPlots[0][0].clear()
            #     AllPtPlots[0][1].clear()

            #     MeanPlots[1][0].set_xlim(0,j+1)
            #     MeanPlots[1][0].set_ylim(allptminmax[0][0],allptminmax[0][1])
            #     vap_axis[1].set_ylim(vapor_pres(allptminmax[0][0]),vapor_pres(allptminmax[0][1]))
            #     MeanPlots[1][0].set_title('ColdFinger Means, '+self.date)
            #     MeanPlots[1][1].set_xlim(0,j)
            #     MeanPlots[1][1].set_ylim(allptminmax[1][0],allptminmax[1][1])
            #     MeanPlots[1][1].set_title('MainCell Means, '+self.date)
            #     MeanPlots[0][0].savefig(self.work_folder+'\\ColdTAvg.png')
            #     MeanPlots[0][1].savefig(self.work_folder+'\\HotTAvg.png')
            #     MeanPlots[0][0].clear()
            #     MeanPlots[0][1].clear()

            #     MeanPlotsTimeScale[1][0].set_xlim(np.floor(min(times)),np.ceil(max(times)))
            #     MeanPlotsTimeScale[1][0].set_ylim(allptminmax[0][0],allptminmax[0][1])
            #     vap_axis[2].set_ylim(vapor_pres(allptminmax[0][0]),vapor_pres(allptminmax[0][1]))
            #     MeanPlotsTimeScale[1][0].set_title('ColdFinger Means vs time, '+self.date)
            #     MeanPlotsTimeScale[1][1].set_xlim(np.floor(min(times)),np.ceil(max(times)))
            #     MeanPlotsTimeScale[1][1].set_ylim(allptminmax[1][0],allptminmax[1][1])
            #     MeanPlotsTimeScale[1][1].set_title('MainCell Means vs time, '+self.date)
            #     MeanPlotsTimeScale[0][0].savefig(self.work_folder+'\\ColdTime.png')
            #     MeanPlotsTimeScale[0][1].savefig(self.work_folder+'\\HotTime.png')
            #     MeanPlotsTimeScale[0][0].clear()
            #     MeanPlotsTimeScale[0][1].clear()

            #     self.plots.update_working_dir(self.folderpath)
            #     self.plots.update_all_imgs()
            #     self.window.title(self.date)
            #     # except:
            #     #     print('Not all valid data!\n select different folder!')
            #     plt.close('all')

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
                self.functions[i] = lambda w,a,p0,h1,h2,mv,T,gamma,base: p0*(1+h1*w+h2*w**2)*np.exp(-a*((w-mv+abs_freq[0])/10**6)*(voigt(w,hyp_weights[0],mv,np.sqrt(T+273.15)*k1*abs_freq[0],gamma)+
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
                self.functions[i] = lambda w,a,p0,h1,mv,T,gamma,base: (p0+h1*w)*np.exp(-a*((w-mv+abs_freq[0])/10**6)*(voigt(w,hyp_weights2[0],mv,np.sqrt(T+273.15)*k1*abs_freq2[0],gamma)+
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
            if i == 1:
                np.savetxt(r'C:\Users\Wolfwalker\Documents\git\6s7p\temp1.csv',self.plotting_freq[1])
            
    def calculate_residuals(self):
        for i in [0,1]:
            self.resid[i] = self.functions[i](self.plotting_freq[i],*self.fitted_params[i]) - self.scaledT[i][self.fit_rng[i][0]:self.fit_rng[i][1]]
            plt.scatter(self.plotting_freq[i],self.scaledT[i][self.fit_rng[i][0]:self.fit_rng[i][1]])
            plt.plot(self.plotting_freq[i],self.functions[i](self.plotting_freq[i],*self.fitted_params[i]),'-r')
            plt.show()
        
    def open_file_dialog(self):
        temporary = filedialog.askdirectory(
            initialdir="/",  # Optional: set initial directory
            title="Select a folder",
            # filetypes=(("Text files", "*.txt"), ("All files", "*.*")) # Optional: filter file types
        )
        self.folderpath = temporary
        if self.folderpath!='':
            date = self.folderpath[:self.folderpath.rfind('/')]
            if os.path.exists(date+r'\Fine.txt'):
                file = open(date+r'\Fine.txt','r')
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
	