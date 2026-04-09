import os as os
import numpy as np
from numpy.polynomial import Polynomial as poly
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
from numpy import pi as pi
from PIL import Image, ImageTk

Day_folder = 'test'
class plots:
    def __init__(self,window,default_path = r".\Picture_template.png", plot_w = 500, plot_h = 300):
        default_img = Image.open(default_path)
        resized_default = default_img.resize((plot_w, plot_h), Image.LANCZOS)
        #Save variables for reference
        self.window = window
        self.plot_w = plot_w
        self.plot_h = plot_h
        #labels to ref plot indices
        self.plotslabs=['TAvg','TAll']
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


class TempAnalysis:
    def __init__(self,window,img_scale=1):
        self.folderpath = ''
        self.folderpath_tkvar = tk.StringVar()
        self.window = window
        self.plots = plots(window=window,plot_w=int(500*img_scale),plot_h=int(300*img_scale))

    def print_path(self):
        print(self.folderpath)

    def checkforanalysis(self):
        months = ['Apr','Mar','Sep','Oct','Nov','Aug']
        found = -1
        if not self.folderpath == '':
            contents = os.listdir(path = self.folderpath)
            for i,mon in enumerate(months):
                found = self.folderpath.find(mon) + 1
                if found:
                    self.date = self.folderpath[found-1:found+10]
            if 'TemperaturePlots' in contents:
                #Then its been run for this day already
                self.plots.update_working_dir(self.folderpath)
                self.plots.update_all_imgs()

            else:
                self.work_folder = self.folderpath+'\\TemperaturePlots'
                os.mkdir(self.work_folder)
                colors = ['black','dodgerblue','crimson']
                #Need to run temperatures analysis
                # try:
                MeanPlots = [[0,0],[0,0]]
                MeanPlots[0][0] = plt.figure()
                MeanPlots[0][1] = plt.figure()
                MeanPlots[1][0] = MeanPlots[0][0].add_axes([0.1,0.1,0.8,0.8])
                MeanPlots[1][1] = MeanPlots[0][1].add_axes([0.1,0.1,0.8,0.8])
                meanplotdat = [[0,0],[0,0]]

                AllPtPlots = [[0,0],[0,0]]
                AllPtPlots[0][0] = plt.figure()
                AllPtPlots[0][1] = plt.figure()
                AllPtPlots[1][0] = AllPtPlots[0][0].add_axes([0.1,0.1,0.8,0.8])
                AllPtPlots[1][1] = AllPtPlots[0][1].add_axes([0.1,0.1,0.8,0.8])

                allptminmax = [[1000,-1000],[1000,-1000]]
                tempminmac = [[0,0],[0,0]]
                print(contents)
                temp = len(contents)
                for num in range(temp):
                    if ('.tsv' in contents[temp-num-1]) or ('xlsx' in contents[temp-num-1]):
                        contents.pop(temp-num-1)
                print(contents)
                for j,run in enumerate(contents):
                    temp=np.loadtxt(self.folderpath+'\\'+run+'\\TemperatureV2.csv',delimiter=',')
                    tempminmac[0][0] = np.min(temp[:,:3])
                    tempminmac[0][1] = np.max(temp[:,:3])
                    tempminmac[1][0] = np.min(temp[:,3:])
                    tempminmac[1][1] = np.max(temp[:,3:])
                    for k in range(2):
                        if tempminmac[k][0] < allptminmax[k][0]:
                            allptminmax[k][0] = tempminmac[k][0]
                        if tempminmac[k][1] > allptminmax[k][1]:
                            allptminmax[k][1] = tempminmac[k][1]
                    #Assume for temperatureV2 for now
                    shape = temp.shape[0]
                    xallPt = np.linspace(j,j+1/3,shape)
                    for i in range(3):
                        meanplotdat[0][0] = np.mean(temp[:,i])
                        meanplotdat[0][1] = np.std(temp[:,i])/np.sqrt(shape)
                        meanplotdat[1][0] = np.mean(temp[:,i+3])
                        meanplotdat[1][1] = np.std(temp[:,i+3])/np.sqrt(shape)
                        #vline
                        MeanPlots[1][0].plot([j+(1+2*i)/6,j+(1+2*i)/6],[meanplotdat[0][0]-meanplotdat[0][1],meanplotdat[0][0]+meanplotdat[0][1]],color=colors[i])
                        MeanPlots[1][1].plot([j+(1+2*i)/6,j+(1+2*i)/6],[meanplotdat[1][0]-meanplotdat[1][1],meanplotdat[1][0]+meanplotdat[1][1]],color=colors[i])
                        #hline
                        MeanPlots[1][0].plot([j+i/3,j+(i+1)/3],[meanplotdat[0][0],meanplotdat[0][0]],color=colors[i])
                        MeanPlots[1][1].plot([j+i/3,j+(i+1)/3],[meanplotdat[1][0],meanplotdat[1][0]],color=colors[i])

                        AllPtPlots[1][0].plot(xallPt+i/3,temp[:,i],color=colors[i])
                        AllPtPlots[1][1].plot(xallPt+i/3,temp[:,i+3],color=colors[i])
                AllPtPlots[1][0].set_xlim(0,j)
                AllPtPlots[1][0].set_ylim(allptminmax[0][0],allptminmax[0][1])
                AllPtPlots[1][0].set_title('ColdFinger All Points, '+self.date)
                AllPtPlots[1][1].set_xlim(0,j)
                AllPtPlots[1][1].set_ylim(allptminmax[1][0],allptminmax[1][1])
                AllPtPlots[1][1].set_title('MainCell All Points, '+self.date)
                AllPtPlots[0][0].savefig(self.work_folder+'\\ColdTAll.png')
                AllPtPlots[0][1].savefig(self.work_folder+'\\HotTAll.png')
                AllPtPlots[0][0].clear()
                AllPtPlots[0][1].clear()

                MeanPlots[1][0].set_xlim(0,j)
                MeanPlots[1][0].set_ylim(allptminmax[0][0],allptminmax[0][1])
                MeanPlots[1][0].set_title('ColdFinger Means, '+self.date)
                MeanPlots[1][1].set_xlim(0,j)
                MeanPlots[1][1].set_ylim(allptminmax[1][0],allptminmax[1][1])
                MeanPlots[1][1].set_title('MainCell Means, '+self.date)
                MeanPlots[0][0].savefig(self.work_folder+'\\ColdTAvg.png')
                MeanPlots[0][1].savefig(self.work_folder+'\\HotTAvg.png')
                MeanPlots[0][0].clear()
                MeanPlots[0][1].clear()
                self.plots.update_working_dir(self.folderpath)
                self.plots.update_all_imgs()
                self.window.title(self.date)
                # except:
                #     print('Not all valid data!\n select different folder!')
                



    def open_file_dialog(self):
        temporary = filedialog.askdirectory(
            initialdir="/",  # Optional: set initial directory
            title="Select a folder",
            # filetypes=(("Text files", "*.txt"), ("All files", "*.*")) # Optional: filter file types
        )
        self.folderpath = temporary
        if self.folderpath!='':
            self.checkforanalysis()

first = True
scale = 1.2
if __name__ == '__main__':
    root = tk.Tk()
    if first:
        first = False
        analysis = TempAnalysis(window=root,img_scale=scale)
        root.title("Temperature Checking")

        open_button = ttk.Button(root, text="Data Folder", command= analysis.open_file_dialog)
        open_button.grid(column=0,row=1)
        
        open_button5 = ttk.Button(root, text="Close", command= exit)
        open_button5.grid(column=1,row=1)
        
        # workingdir_txt = tk.StringVar()
        # workingdir_txt.set('hello')
        # workingdir = ttk.Label(root, textvariable=workingdir_txt)
        # workingdir.grid(column=3, row=11,columnspan=3, sticky="nsew")
    root.mainloop()
	