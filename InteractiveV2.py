import os as os
import numpy as np
from numpy.polynomial import Polynomial as poly
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import TestingDataType
from numpy import pi as pi
from plotClass import plots
from scipy.special import wofz as wofz


class window:
    def __init__(self,window,default_path = r".\Picture_template.png", plot_w = 500, plot_h = 300):
        default_img = Image.open(default_path)
        ent_wdth = 20
        resized_default = default_img.resize((plot_w, plot_h), Image.LANCZOS)
        #Save variables for reference
        self.window = window
        self.plot_w = plot_w
        self.plot_h = plot_h
        #labels to ref plot indices
        self.plotslabs = [['Tavg','Pavg','HotAvg','scaledHot','scaledT','FittedScan','FittedScanResid'],['ogbeat','filteredbeat','fitted_beat','unscaledresiduals','VapPres']]
        self.scan=['456','894']
        self.day_fold = ''
        self.window_manager={}
        i=-1
        for scan in self.scan:
            i+=1
            self.window_manager[scan]={}
            self.window_manager[scan]['Notes'] = []
            self.window_manager[scan]['Imgs'] = [{},{}]
            self.window_manager[scan]['entries'] = {'fit_rng':{'entry':[ttk.Entry(self.window,width=ent_wdth), ttk.Entry(self.window,width=ent_wdth)],'val':[tk.StringVar(value="0"), tk.StringVar(value="8000")]}}
            self.window_manager[scan]['entries']['fit_rng']['entry'][0].configure(textvariable=self.window_manager[scan]['entries']['fit_rng']['val'][0])
            self.window_manager[scan]['entries']['fit_rng']['entry'][0].grid(column=1,row=1+2*int(scan=='894'))
            self.window_manager[scan]['entries']['fit_rng']['entry'][1].configure(textvariable=self.window_manager[scan]['entries']['fit_rng']['val'][1])
            self.window_manager[scan]['entries']['fit_rng']['entry'][1].grid(column=2,row=1+2*int(scan=='894'))

            self.window_manager[scan]['entries']['beat_min']={'entry':[ttk.Entry(self.window,width=ent_wdth)],'val':[tk.StringVar(value="0")]}
            self.window_manager[scan]['entries']['beat_min']['entry'][0].configure(textvariable=self.window_manager[scan]['entries']['fit_rng']['val'][0])
            self.window_manager[scan]['entries']['beat_min']['entry'][0].grid(column=5,row=1+2*int(scan=='894'))

            self.window_manager[scan]['labels'] = [ttk.Label(self.window,text='Fit Range'),ttk.Label(self.window,text='Beat min')]
            self.window_manager[scan]['labels'][0].grid(column=0,row=1+2*int(scan=='894'))
            self.window_manager[scan]['labels'][1].grid(column=4,row=1+2*int(scan=='894'))
            for j in [0,1]:
                self.window_manager[scan]['Notes'].append(ttk.Notebook(window))
                self.window_manager[scan]['Notes'][-1].grid(column=j*4,row=int(scan=='894')*2,columnspan=3, sticky="nsew")
                for name in self.plotslabs[j]:
                    self.window_manager[scan]['Imgs'][j][name] = {}
                    self.window_manager[scan]['Imgs'][j][name]['TkImg']=ImageTk.PhotoImage(resized_default.copy())
                    self.window_manager[scan]['Imgs'][j][name]['Label']= tk.Label(self.window_manager[scan]['Notes'][-1],image=self.window_manager[scan]['Imgs'][j][name]['TkImg'])
                    self.window_manager[scan]['Imgs'][j][name]['Label'].image = self.window_manager[scan]['Imgs'][j][name]['TkImg']
                    self.window_manager[scan]['Imgs'][j][name]['Label'].pack()
                    self.window_manager[scan]['Notes'][-1].add(self.window_manager[scan]['Imgs'][j][name]['Label'],text=scan+' '+name)

        close = ttk.Button(self.window, text="Close", command= exit)
        close.grid(column=3,row=10)

        workingdir_txt = tk.StringVar()
        workingdir_txt.set('hello')
        workingdir = ttk.Label(self.window, textvariable=workingdir_txt)
        workingdir.grid(column=3, row=11,columnspan=3, sticky="nsew")
        # self.window_manager = {'Cold':{'Note':ttk.Notebook(window)},'Hot':{'Note':ttk.Notebook(window)}}
        # self.window_manager['Cold']['Note'].grid(column=0,row=0,columnspan=1, sticky="nsew")
        # self.window_manager['Hot']['Note'].grid(column=1,row=0,columnspan=1, sticky="nsew")
        # self.window_manager['Cold']['']    -
    
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
        for scan in self.scan:
            for name in self.plotslabs:
                if name != 'TBD':
                    plot_path = self.fold + '\\' + scan +name + '.png'
                    temp = Image.open(plot_path)
                    resized_temp = temp.resize((self.plot_w, self.plot_h), Image.LANCZOS)
                    self.window_manager[scan]['Imgs'][name]['TkImg'] = ImageTk.PhotoImage(resized_temp)
                    self.change_Label_image(self.window_manager[scan]['Imgs'][name]['TkImg'],self.window_manager[scan]['Imgs'][name]['Label'])

class analysisV2:
    #V2 includes the plots from simultaneous hot cell meas
    pass 


first = True
img_scale = 1.6
if __name__ == '__main__':
    root = tk.Tk()
    if first:
        first = False
        test = window(root,plot_w=int(500*img_scale),plot_h=int(300*img_scale))
        # analysis = ResidAnalysis(window=root,img_scale=scale)
        # root.title("Residual Calculations")

        # open_button = ttk.Button(root, text="Data Folder", command= analysis.open_file_dialog)
        # open_button.grid(column=0,row=1)
        
        # open_button5 = ttk.Button(root, text="Close", command= exit)
        # open_button5.grid(column=1,row=1)
        
        # workingdir_txt = tk.StringVar()
        # workingdir_txt.set('hello')
        # workingdir = ttk.Label(root, textvariable=workingdir_txt)
        # workingdir.grid(column=3, row=11,columnspan=3, sticky="nsew")
    root.mainloop()