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
        self.scans=['456','894']
        self.day_fold = ''
        self.window_manager={}
        i=-1
        for scan in self.scans:
            i+=1
            self.window_manager[scan]={}
            self.window_manager[scan]['Notes'] = []
            self.window_manager[scan]['dir'] = ''
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

        self.window_manager['clickable']={'456':{'calcTFit':{},'calcBeatFit':{},'show':{}},
                                          '894':{'calcTFit':{},'calcBeatFit':{},'show':{}},
                                          'both':{'open_fold':{},'save':{},'exit':{}}}
        i=-2
        for key1 in self.window_manager['clickable'].keys():
            for key2 in self.window_manager['clickable'][key1].keys():
                self.window_manager['clickable'][key1][key2]['func'] = lambda : print('Pick a data set for analysis!')
                self.window_manager['clickable'][key1][key2]['button'] = ttk.Button(self.window,text=key2,command=self.window_manager['clickable'][key1][key2]['func'])
                if key1 == 'both':
                    i+=2
                    self.window_manager['clickable'][key1][key2]['button'].grid(column=3,row=i)
                else:
                    self.window_manager['clickable'][key1][key2]['button'].configure(text=key1+' '+key2)
                    if key2 == 'show':
                        pass
                    else:
                        self.window_manager['clickable'][key1][key2]['button'].grid(row=1+2*int(key1=='894'),column=3+3*int(key2=='calcBeatFit'))
        self.window_manager['clickable']['both']['exit']['button'].configure(command=exit)
        # close = ttk.Button(self.window, text="Close", command= exit)
        # close.grid(column=3,row=10)

        self.window_manager['work_dir'] = {'path':'Pick Directory','tk_var':tk.StringVar()}
        self.window_manager['work_dir']['tk_var'].set('Pick Directory')
        self.window_manager['work_dir']['lab'] = ttk.Label(self.window, textvariable=self.window_manager['work_dir']['tk_var'])
        self.window_manager['work_dir']['lab'].grid(column=3, row=11,columnspan=3, sticky="nsew")

    def update_work_dir(self,new_par_fold):
        self.window_manager['456']['dir']=new_par_fold+r'\Analysis\456\plots'
        self.window_manager['894']['dir']=new_par_fold+r'\Analysis\894\plots'
    
    def update_image(self,scan,name):
        if scan in self.scans:
            if name in self.plotslabs[0]:temp=1
            elif name in self.plotslabs[1]:temp=2
            else: temp=0
            if temp:
                plot_path = self.window_manager[scan]['dir'] + '\\' + name + '.png'
                temp2 = Image.open(plot_path)
                resized_temp2 = temp2.resize((self.plot_w, self.plot_h), Image.LANCZOS)
                self.window_manager[scan]['Imgs'][temp-1][name]['TkImg'] = ImageTk.PhotoImage(resized_temp2)
                self.window_manager[scan]['Imgs'][temp-1][name]['Label'].configure(image=self.window_manager[scan]['Imgs'][temp-1][name]['TkImg'])
                self.window_manager[scan]['Imgs'][temp-1][name]['Label'].image = self.window_manager[scan]['Imgs'][temp-1][name]['TkImg']
        

    
    # def change_Label_image(self,new,oldlabel):
    # #oldlabel is the label you want to change and
    # #new is new Tkimage to exchange
    #     oldlabel.configure(image=new)
    #     oldlabel.image = new
    
    # def update_all_imgs(self):
    #     for scan in self.scan:
    #         for name in self.plotslabs:
    #             if name != 'TBD':
    #                 plot_path = self.fold + '\\' + scan +name + '.png'
    #                 temp = Image.open(plot_path)
    #                 resized_temp = temp.resize((self.plot_w, self.plot_h), Image.LANCZOS)
    #                 self.window_manager[scan]['Imgs'][name]['TkImg'] = ImageTk.PhotoImage(resized_temp)
    #                 self.change_Label_image(self.window_manager[scan]['Imgs'][name]['TkImg'],self.window_manager[scan]['Imgs'][name]['Label'])

class analysisV2:
    #V2 includes the plots from simultaneous hot cell meas
    def __init__(self,root,img_scale):
        self.root =  root
        self.wind = window(root, plot_w=int(500*img_scale),plot_h=int(300*img_scale))
        # temp = lambda:print('Pick Folder')
        # open_button = ttk.Button(root, text="Data Folder", command= self.open_file_dialog)
        # open_button.grid(column=3,row=0)
        self.folderpath = ''
    
    def checkforanalysis(self):
        if not self.folderpath == '':
            # contents = os.listdir(path = self.folderpath)
            # date = self.folderpath[:self.folderpath.rfind('/')]

            # if os.path.exists(date+r'\Fine.txt'):
            #     file = open(date+r'\Fine.txt','r')
            #     Fine = list(map(int,file.readline().split(',')))
            #     file.close()
            # else:
            #     Fine = [0,0] 

            # file = open(self.folderpath+r'\beatnote_det_f.csv','r')
            # beat_det_f = list(map(float,file.readline().strip().split(',')))
            # file.close()

            # if 'Analysis' in contents:
            #     print('Analysis exists, continue')
            #     self.Temperature = np.loadtxt(self.folderpath+r'\Analysis\TempMeas.csv', delimiter=',')
            #     self.analysis456 = TestingDataType.data(self.folderpath,exists=True,beatnote_det_f=beat_det_f[0]/1000,F=Fine[0])
            #     self.analysis894 = TestingDataType.data(self.folderpath,scan='894',exists=True,beatnote_det_f=beat_det_f[1]/1000,F=Fine[1])
            # else:
            #     print('Analysis does not exist ')
            #     os.mkdir(self.folderpath+r'\Analysis')
            #     os.mkdir(self.folderpath+r'\Analysis\456')
            #     os.mkdir(self.folderpath+r'\Analysis\456\beatnote')
            #     os.mkdir(self.folderpath+r'\Analysis\456\beatnote\original')
            #     os.mkdir(self.folderpath+r'\Analysis\456\beatnote\processed')
            #     os.mkdir(self.folderpath+r'\Analysis\456\fitting')
            #     os.mkdir(self.folderpath+r'\Analysis\456\fitting\original')
            #     os.mkdir(self.folderpath+r'\Analysis\456\fitting\processed')
            #     os.mkdir(self.folderpath+r'\Analysis\456\plots')
            #     os.mkdir(self.folderpath+r'\Analysis\456\entries')
            #     os.mkdir(self.folderpath+r'\Analysis\894')
            #     os.mkdir(self.folderpath+r'\Analysis\894\beatnote')
            #     os.mkdir(self.folderpath+r'\Analysis\894\beatnote\original')
            #     os.mkdir(self.folderpath+r'\Analysis\894\beatnote\processed')
            #     os.mkdir(self.folderpath+r'\Analysis\894\fitting')
            #     os.mkdir(self.folderpath+r'\Analysis\894\fitting\original')
            #     os.mkdir(self.folderpath+r'\Analysis\894\fitting\processed')
            #     os.mkdir(self.folderpath+r'\Analysis\894\plots')
            #     os.mkdir(self.folderpath+r'\Analysis\894\entries')
            #     temp = Image.open(r".\Picture_template.png")
            #     temp.save(self.folderpath+r'\Analysis\456\plots\FittedScan.png')
            #     temp.save(self.folderpath+r'\Analysis\456\plots\FittedScanResid.png')
            #     temp.save(self.folderpath+r'\Analysis\894\plots\FittedScan.png')
            #     temp.save(self.folderpath+r'\Analysis\894\plots\FittedScanResid.png')
            #     f = open(self.analysis456.folder+r'\entries\beat_peak_min.csv','w')
            #     f.write(str(0))
            #     f.close()
            #     f = open(self.analysis894.folder+r'\entries\beat_peak_min.csv','w')
            #     f.write(str(0))
            #     f.close()

            #     temps = TestingDataType.simple_dat_get(self.folderpath +r'\TemperatureV2.csv',0)
            #     self.Temperature = []
            #     for i in range(6):
            #         self.Temperature.append(np.mean(temps[:,i]))
            #     np.savetxt(self.folderpath+r'\Analysis\TempMeas.csv', self.Temperature, delimiter=',')
            #     self.analysis456 = TestingDataType.data(self.folderpath,exists=False,beatnote_det_f=beat_det_f[0]/1000,F=Fine[0])
            #     self.analysis894 = TestingDataType.data(self.folderpath,scan='894',exists=False,beatnote_det_f=beat_det_f[1]/1000,F=Fine[1])

            self.wind.window_manager['work_dir']['tk_var'].set(self.folderpath)
            print('here')
            self.functs[0] = self.test
            self.but.configure(command=self.functs[0])


            
    def test(self):
        print(2+2)

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


first = True
template_image = r".\Picture_template.png"
if os.getlogin() == 'garci868':
        scale = 1.2
else:
    scale = 1.7
if __name__ == '__main__':
    if first:
        root = tk.Tk()
        first = False
        test = analysisV2(root,img_scale=scale)
    root.mainloop()