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

def check_for_analysis(folder,F1,check_against=[]):
    laser = '894'
    # if os.getlogin() == 'garci868':
    #         laser = '894'
    # else:
    #     laser = '456'
    temp = os.listdir(path = folder)
    if 'Analysis' in temp:
        if not (True in list(map(lambda y: y in folder, check_against))):
            #If analysis is in current dir
            #redo analysis if current measurement is not in 456Fitparams
            redo_analysis(folder,F1)
    else:
        check_against = []
        try:
            if laser == '456':
                fits_ind = list(map(lambda y:'456Fitparams' in y,temp)).index(True)
            else:
                fits_ind = list(map(lambda y:'894Fitparams' in y,temp)).index(True)
            file = open(folder+'\\'+temp[fits_ind],'r')
            file.readline()
            for line in file:
                check_against.append(line.split('\t')[0])
            file.close()
        except:
            print('here')
        
        x = list(os.scandir(folder))
        for val in x:
            if val.is_dir():
                check_for_analysis(val.path,F1,check_against)
    
def redo_analysis(par_folder,F1):
    #we know that the analysis exists so we need to check if they've been previously fitted
    if os.path.exists(par_folder+r'\Analysis\456\fitting\processed\fitting_param.csv') and os.path.exists(par_folder+r'\Analysis\894\fitting\processed\fitting_param.csv'):
        # if not os.path.exists(par_folder+'\\redone.txt'):
            #we've fit before
        print(par_folder)
        # if os.getlogin() == 'garci868':
        #     laser = '894'
        # else:
        #     laser = '456'
        # file = open(par_folder+'\\redone2.txt','w+')
        # file.close()
        to_do = ['894']
        for laser in to_do:
            if laser == '456':
                scan = Absorption_calc.data(par_folder,exists=True)
                scan.F1=F1
                scan.set_transition(F1=F1)
            else:
                scan = Absorption_calc.data(par_folder,scan='894',exists=True)
            scan.calculate_beat_fit()
            scan.set_fitting_function()
    
base_dir = os.getcwd()
folders = [r'\BeatnotePostHotCellF1=4',r'\BeatnotePostHotCellF1=4']
# subfolders = [['Jul07,2026','Jul08,2026', 'Jul09,2026', 'Jul13,2026', 'Jul14,2026', 'Jul15,2026', 'Jul16,2026', 'Jul17,2026'],['Jul20,2026', 'Jul21,2026', 'Jun25,2026', 'Jun26,2026', 'Jun29,2026','Jun30,2026', 'May22,2026', 'May28,2026', 'May29,2026']]


F1s = [3,4]

if os.getlogin() == 'garci868':
    base_folder = base_dir+ r'\BeatnotePostHotCell'
    F1 = 3
    check_for_analysis(base_folder,F1,check_against=[])
else:
    base_folder = base_dir+ r'\BeatnotePostHotCellF1=4'
    F1 = 4
# if __name__ == '__main__':
#     if os.getlogin() == 'garci868':
#         pass
#         for sub in subfolders[0]:
#             check_for_analysis(base_dir+folders[0]+'\\'+sub,F1s[1])
#     else:

#         for sub in subfolders[1]:
#             check_for_analysis(base_dir+folders[1]+'\\'+sub,F1s[1])