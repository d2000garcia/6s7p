import os as os
import numpy as np
from numpy.polynomial import Polynomial as poly
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import Absorption_calc
from numpy import pi as pi
from scipy.special import wofz as wofz

class analysisV2:
    #V2 includes the plots from simultaneous hot cell meas
    def __init__(self):
        self.folderpath = ''
        self.analysis = [0,0]
        self.beatmin = [0,0]
    
    def checkforanalysis(self):
        if not self.folderpath == '':
            contents = os.listdir(path = self.folderpath)
            date = self.folderpath[:self.folderpath.rfind('/')]
            if os.path.exists(date+r'\Fine.txt'):
                file = open(date+r'\Fine.txt','r')
                Fine = list(map(int,file.readline().split(',')))
                file.close()
            else:
                Fine = [0,0] 
            if 'Analysis' in contents:
                print('Analysis exists, continue')
                self.analysis[0] = Absorption_calc.data(self.folderpath,exists=True)
                self.analysis[1] = Absorption_calc.data(self.folderpath,scan='894',exists=True)
            else:
                print('Analysis does not exist ')
                os.mkdir(self.folderpath+r'\Analysis')
                os.mkdir(self.folderpath+r'\Analysis\456')
                os.mkdir(self.folderpath+r'\Analysis\456\beatnote')
                os.mkdir(self.folderpath+r'\Analysis\456\beatnote\original')
                os.mkdir(self.folderpath+r'\Analysis\456\beatnote\processed')
                os.mkdir(self.folderpath+r'\Analysis\456\fitting')
                os.mkdir(self.folderpath+r'\Analysis\456\fitting\original')
                os.mkdir(self.folderpath+r'\Analysis\456\fitting\processed')
                os.mkdir(self.folderpath+r'\Analysis\456\plots')
                os.mkdir(self.folderpath+r'\Analysis\456\entries')
                os.mkdir(self.folderpath+r'\Analysis\894')
                os.mkdir(self.folderpath+r'\Analysis\894\beatnote')
                os.mkdir(self.folderpath+r'\Analysis\894\beatnote\original')
                os.mkdir(self.folderpath+r'\Analysis\894\beatnote\processed')
                os.mkdir(self.folderpath+r'\Analysis\894\fitting')
                os.mkdir(self.folderpath+r'\Analysis\894\fitting\original')
                os.mkdir(self.folderpath+r'\Analysis\894\fitting\processed')
                os.mkdir(self.folderpath+r'\Analysis\894\plots')
                os.mkdir(self.folderpath+r'\Analysis\894\entries')
                temp = Image.open(r".\Picture_template.png")
                
                temp.save(self.folderpath+r'\Analysis\456\plots\FittedScan.png')
                temp.save(self.folderpath+r'\Analysis\456\plots\FittedScanResid.png')
                temp.save(self.folderpath+r'\Analysis\456\plots\fitted_beat.png')
                temp.save(self.folderpath+r'\Analysis\456\plots\unscaledresiduals.png')

                temp.save(self.folderpath+r'\Analysis\894\plots\FittedScan.png')
                temp.save(self.folderpath+r'\Analysis\894\plots\FittedScanResid.png')
                temp.save(self.folderpath+r'\Analysis\894\plots\fitted_beat.png')
                temp.save(self.folderpath+r'\Analysis\894\plots\unscaledresiduals.png')
                np.savetxt(self.folderpath+'\\Analysis\\456\\entries\\beat_peak_min.csv',[0],delimiter=',')
                np.savetxt(self.folderpath+'\\Analysis\\894\\entries\\beat_peak_min.csv',[0],delimiter=',')
                np.savetxt(self.folderpath+'\\Analysis\\456\\entries\\fit_rng.csv',[0,8000],delimiter=',',fmt='%i')
                np.savetxt(self.folderpath+'\\Analysis\\894\\entries\\fit_rng.csv',[0,8000],delimiter=',',fmt='%i')
                self.analysis[0]=Absorption_calc.data(self.folderpath,exists=False)
                self.analysis[1]=Absorption_calc.data(self.folderpath,scan='894',exists=False)

            for s in ['456','894']:
                temp = np.loadtxt(self.folderpath+'\\Analysis\\'+s+'\\entries\\beat_peak_min.csv',delimiter=',').tolist()
                temp2 = np.loadtxt(self.folderpath+'\\Analysis\\'+s+'\\entries\\fit_rng.csv',delimiter=',',dtype=int)

    def calculateTFit(self,scan):
        if self.analysis[int(scan!='456')].isbeatfitted:
            self.analysis[int(scan!='456')].set_fitting_function()
            self.wind.update_image(scan,'FittedScan')
            self.wind.update_image(scan,'FittedScanResid')
        else:print("Fit "+ scan +" beatnote first!")

    def calculateBeatFit(self,scan):
        temp = float(self.wind.window_manager[scan]['entries']['beat_min']['val'][0].get())
        temp2 = float(np.loadtxt(self.folderpath+'\\Analysis\\'+scan+'\\entries\\beat_peak_min.csv',delimiter=','))
        # print(temp)
        # print(temp2)
        if temp != temp2:
            np.savetxt(self.folderpath+'\\Analysis\\'+scan+'\\entries\\beat_peak_min.csv',[temp],delimiter=',')
        temp = [int(self.wind.window_manager[scan]['entries']['fit_rng']['val'][0].get())]
        temp.append(int(self.wind.window_manager[scan]['entries']['fit_rng']['val'][1].get()))
        temp2 = np.loadtxt(self.folderpath+'\\Analysis\\'+scan+'\\entries\\fit_rng.csv',delimiter=',',dtype=int)
        if (temp[0] != int(temp2[0])) or (temp[1] != int(temp2[1])):
            np.savetxt(self.folderpath+'\\Analysis\\'+scan+'\\entries\\fit_rng.csv',temp,delimiter=',',fmt='%i')
        self.analysis[int(scan!='456')].beat_rng = temp.copy()
        self.analysis[int(scan!='456')].filter_beatnote()
        to_update = ['scaledH','scaledT','filteredbeat','fitted_beat','unscaledresiduals']
        try:
            self.analysis[int(scan!='456')].calculate_beat_fit()
            plot = True
        except:print('Not able to fitbeatnote')
        for pic in to_update:
            self.wind.update_image(scan,pic)

    def open_file_dialog(self):
        temporary = filedialog.askdirectory(
            initialdir="/",  # Optional: set initial directory
            title="Select a folder",
            # filetypes=(("Text files", "*.txt"), ("All files", "*.*")) # Optional: filter file types
        )
        if temporary:
            self.folderpath = temporary
            date_time = self.folderpath[self.folderpath.rfind('/')+1:]
            self.root.title(date_time + ' Fiting Analysis')
            print(f"Selected folder: {self.folderpath}")
            self.checkforanalysis()

def check_for_analysis(folder,F1):
    if 'Analysis' in os.listdir(path = folder):
        redo_analysis(folder,F1)
    else:
        x = list(os.scandir(folder))
        for val in x:
            if val.is_dir():
                check_for_analysis(val.path,F1)
    
def redo_analysis(par_folder,F1):
    #we know that the analysis exists so we need to check if they've been previously fitted
    if os.path.exists(par_folder+r'\Analysis\456\fitting\processed\fitting_param.csv') and os.path.exists(par_folder+r'\Analysis\894\fitting\processed\fitting_param.csv'):
        if not os.path.exists(par_folder+'\\redone.txt'):
            #we've fit before
            did = True
            print(par_folder)
            file = open(par_folder+'\\redone.txt','w+')
            file.close()
            scan = Absorption_calc.data(par_folder,exists=True)
            scan.F1=F1
            scan.set_transition(F1=F1)
            scan.set_fitting_function()
    


base_dir = os.getcwd()

base_folder = base_dir+ r'\BeatnotePostHotCell\Half1'
F1 = 4
if __name__ == '__main__':
    check_for_analysis(base_folder,F1)