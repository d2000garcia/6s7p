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

from plotClass import plots


class analysis:
    def __init__(self):
        self.folderpath = ''
        self.folderpath_tkvar = tk.StringVar()

    def print_path(self):
        print(self.folderpath)
    
    def checkforanalysis(self):
        if not self.folderpath == '':
            contents = os.listdir(path = self.folderpath)
            if 'Analysis' in contents:
                print('Analysis exists, continue')
                self.analysis456 = TestingDataType.data(self.folderpath,exists=True)
                self.analysis894 = TestingDataType.data(self.folderpath,scan='894',exists=True)
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
                temp.save(self.folderpath+r'\Analysis\894\plots\FittedScan.png')
                temp.save(self.folderpath+r'\Analysis\894\plots\FittedScanResid.png')
                self.analysis456 = TestingDataType.data(self.folderpath,exists=False)
                self.analysis894 = TestingDataType.data(self.folderpath,scan='894',exists=False)
            self.Temperature = TestingDataType.simple_dat_get(self.folderpath+r'\Temperature.csv')[0][0]
            # self.folderpath_tkvar.set(self.folderpath)
            update_path_label(self.folderpath)
    
    def record_fits(self):
        date = self.folderpath[self.folderpath.rfind('/')+1:]
        lines = {}
        if self.analysis456.fitted and self.analysis894.fitted:
            data = [date, str(self.Temperature), str(self.analysis456.alpha),str(self.analysis456.alph_err),str(self.analysis894.alpha),str(self.analysis894.alph_err)]
            print(data)
            file = open(r'.\Fits.tsv',"r")
            file.readline()
            for line in file:
                line = line.strip().split('\t')
                lines[line[0]]=line
            file.close()
            lines[date] = data
            order = list(lines.keys())
            order.sort()
            file = open(r'.\Fits.tsv',"w")
            file.write('Date\tTemp\t456alph\t456err\t894alph\t894err\n')
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
            print('Saved')
                


        

def open_file_dialog(analysis_dat,labels,entries,plots456,plots894,switchlabelsat=5):
    temporary = filedialog.askdirectory(
        initialdir="/",  # Optional: set initial directory
        title="Select a folder",
        # filetypes=(("Text files", "*.txt"), ("All files", "*.*")) # Optional: filter file types
    )
    analysis_dat.folderpath = temporary
    if analysis_dat.folderpath:
        print(f"Selected folder: {analysis_dat.folderpath}")
        analysis_dat.checkforanalysis()
        plots456.update_working_dir(analysis_dat.folderpath)
        plots894.update_working_dir(analysis_dat.folderpath)
        for i, lab in enumerate(plots456.plotslabs):
            plots456.update_image([lab])
            plots894.update_image([lab])
            if i < switchlabelsat:
                change_Label_image(labels[0][i], plots456.plots[i])
                change_Label_image(labels[2][i], plots894.plots[i])
            else:
                change_Label_image(labels[1][i-switchlabelsat], plots456.plots[i])
                change_Label_image(labels[3][i-switchlabelsat], plots894.plots[i])
        for i in [0,1]:
            entries[1][0][i].set(str(analysis_dat.analysis456.back_rngs[0][i]))

            entries[1][0][2+i].set(str(analysis_dat.analysis456.back_rngs[1][i]))
            entries[1][2][i].set(str(analysis_dat.analysis894.back_rngs[0][i]))
            entries[1][2][2+i].set(str(analysis_dat.analysis894.back_rngs[1][i]))

            entries[1][1][i].set(str(analysis_dat.analysis456.beat_rng[i]))
            entries[1][3][i].set(str(analysis_dat.analysis894.beat_rng[i]))

            for i in range(len(entries[0])):
                for j in range(len(entries[0][i])):
                    entries[0][i][j].configure(textvariable=entries[1][i][j])


def update_path_label(text):
    workingdir_txt.set(value=text)

def change_Label_image(oldlabel,new):
    #oldlabel is the label you want to change and
    #new is new Tkimage to exchange
    oldlabel.configure(image=new)
    oldlabel.image = new

def recalculate456T(analysis_dat, labels, entries, plots_grp,switchlabelsat=5):
    for i in [0,1]:
            analysis_dat.analysis456.back_rngs[0][i] = int(entries[1][0][i].get())
            analysis_dat.analysis456.back_rngs[1][i] = int(entries[1][0][2+i].get())
    analysis_dat.analysis456.calculate_T_shift()
    plots_grp.update_image(['scaledT'])
    tochange = [2]
    for i in tochange:
        change_Label_image(labels[0][i], plots_grp.plots[i])

def recalculate456beat(analysis_dat, labels, entries, plots_grp,switchlabelsat=5):
    analysis_dat.analysis456.beat_rng[0] = int(entries[1][1][0].get())
    analysis_dat.analysis456.beat_rng[1] = int(entries[1][1][1].get())
    analysis_dat.analysis456.calculate_beat_fit()
    plots_grp.update_image(['filteredbeat','fitted_beat','unscaledresiduals','ScaledResiduals','scaledT'])
    tochange = [6,7,8,9]
    for i in tochange:
        change_Label_image(labels[1][i-switchlabelsat], plots_grp.plots[i])
    tochange = [2]
    for i in tochange:
        change_Label_image(labels[0][i], plots_grp.plots[i])

def recalculate894T(analysis_dat, labels, entries, plots_grp,switchlabelsat=5):
    for i in [0,1]:
            analysis_dat.analysis894.back_rngs[0][i] = int(entries[1][2][i].get())
            analysis_dat.analysis894.back_rngs[1][i] = int(entries[1][2][2+i].get())
    analysis_dat.analysis894.calculate_T_shift()
    plots_grp.update_image(['scaledT'])
    tochange = [2]
    for i in tochange:
        change_Label_image(labels[2][i], plots_grp.plots[i])

def recalculate894beat(analysis_dat, labels, entries, plots_grp,switchlabelsat=5):
    analysis_dat.analysis894.beat_rng[0] = int(entries[1][3][0].get())
    analysis_dat.analysis894.beat_rng[1] = int(entries[1][3][1].get())
    analysis_dat.analysis894.calculate_beat_fit()
    plots_grp.update_image(['filteredbeat','fitted_beat','unscaledresiduals','ScaledResiduals','scaledT'])
    tochange = [6,7,8,9]
    for i in tochange:
        change_Label_image(labels[3][i-switchlabelsat], plots_grp.plots[i])
    tochange = [2]
    for i in tochange:
        change_Label_image(labels[2][i], plots_grp.plots[i])

def calculate_abs_fit_456(analysis_dat,labels,plots_grp):
    analysis_dat.analysis456.set_fitting_function()
    plots_grp.update_image(['FittedScan','FittedScanResid'])
    tochange = [3,4]
    for i in tochange:
        change_Label_image(labels[0][i], plots_grp.plots[i])
def calculate_abs_fit_894(analysis_dat,labels,plots_grp):
    analysis_dat.analysis894.set_fitting_function()
    plots_grp.update_image(['FittedScan','FittedScanResid'])
    tochange = [3,4]
    for i in tochange:
        change_Label_image(labels[2][i], plots_grp.plots[i])

def calculate_ratio(analysis_dat):
    if analysis_dat.analysis894.fitted and analysis_dat.analysis456.fitted:
        print(analysis_dat.analysis456.alpha/analysis_dat.analysis894.fitted)

def record_fits(analysis_dat):
    analysis_dat.record_fits()

first = True
template_image = r".\Picture_template.png"
switchlabelsat=5
if __name__ == '__main__':
    root = tk.Tk()
    if first:
        first = False
        folder = analysis()
        plot_sets= [plots(scan='456'),plots(scan='894')]
        # plots456 = plots(scan='456')
        # plots894 = plots(scan='894')
        root.title("Geometry Calculation")

        notebooks= [ttk.Notebook(root),ttk.Notebook(root),ttk.Notebook(root),ttk.Notebook(root)]
        #notebooks = [456T,456Beat,894T,894Beat]
        notebooks[0].grid(column=0,row=0,columnspan=3, sticky="nsew")
        notebooks[1].grid(column=4,row=0,columnspan=3, sticky="nsew")
        notebooks[2].grid(column=0,row=4,columnspan=3, sticky="nsew")
        notebooks[3].grid(column=4,row=4,columnspan=3, sticky="nsew")


        entry_lbl = [ttk.Label(root, text='456  Background Linear Fit Ranges'), ttk.Label(root, text='456 Beatnote Range')]
        entry_lbl.append(ttk.Label(root, text='Left data group'))
        entry_lbl.append(ttk.Label(root, text='Right data group'))
        entry_lbl.append(ttk.Label(root, text='894 Background Linear Fit Rngs'))
        entry_lbl.append(ttk.Label(root, text='894 Beatnote Range'))
        entry_lbl.append(ttk.Label(root, text='Left data group'))
        entry_lbl.append(ttk.Label(root, text='Right data group'))
        entry_lbl[0].grid(column=1,row=1,columnspan=2, sticky="nsew")
        entry_lbl[1].grid(column=5,row=1,columnspan=2, sticky="nsew")
        entry_lbl[2].grid(column=0,row=2)
        entry_lbl[3].grid(column=0,row=3)
        entry_lbl[4].grid(column=1,row=5,columnspan=2, sticky="nsew")
        entry_lbl[5].grid(column=5,row=5,columnspan=2, sticky="nsew")
        entry_lbl[6].grid(column=0,row=6)
        entry_lbl[7].grid(column=0,row=7)

        ent_wdth = 20
        entries = [[[ttk.Entry(root,width=ent_wdth), ttk.Entry(root,width=ent_wdth), ttk.Entry(root,width=ent_wdth), ttk.Entry(root,width=ent_wdth)],[ttk.Entry(root,width=ent_wdth), ttk.Entry(root,width=ent_wdth)]]]
        entries.append([[tk.StringVar(value="0"), tk.StringVar(value="0"), tk.StringVar(value="0"), tk.StringVar(value="0")],[tk.StringVar(value="0"), tk.StringVar(value="0")]])
        entries[0].append([ttk.Entry(root,width=ent_wdth), ttk.Entry(root,width=ent_wdth), ttk.Entry(root,width=ent_wdth), ttk.Entry(root,width=ent_wdth)])
        entries[1].append([tk.StringVar(value="0"), tk.StringVar(value="0"), tk.StringVar(value="0"), tk.StringVar(value="0")])
        entries[0].append([ttk.Entry(root,width=ent_wdth), ttk.Entry(root,width=ent_wdth)])
        entries[1].append([tk.StringVar(value="0"), tk.StringVar(value="0")])
        for i in range(len(entries[0])):
            for j in range(len(entries[0][i])):
                entries[0][i][j].configure(textvariable=entries[1][i][j])

        
        temp = 0
        for j in range(4):
            if j>1:
                temp=1
            else:
                entries[0][1][j].grid(column=5+j,row=2+temp)
                entries[0][3][j].grid(column=5+j,row=6+temp)
            entries[0][0][j].grid(column=j%2+1,row=2+temp)
            entries[0][2][j].grid(column=j%2+1,row=6+temp)

        #entries[0 & 2] = entries for background linear fitting of scaled T for 456 and 894 respectively
        #entries [1 & 3] = entries for beatnote range to include in the fitting

        labels = [[],[],[],[]]
        #labels = [labels456T,labels456Beat, labels894T,labels894Beat]
        
        for i, lab in enumerate(plot_sets[0].plotslabs):
            #Makes labels and includes plots in them
            for j in [0,2]:
                temp = ((j+1)%3+1)%2
                k=0
                if not i < switchlabelsat:
                    k=1
                labels[j+k].append(tk.Label(notebooks[j+k],image=plot_sets[temp].plots[i]))
                labels[j+k][-1].image = plot_sets[temp].plots[i]
                labels[j+k][-1].pack()
                notebooks[j+k].add(labels[j+k][-1],text=plot_sets[temp].scan+' '+lab)
        entry = ttk.Entry(root,width=ent_wdth)
        open_button = ttk.Button(root, text="Data Folder", command= lambda: open_file_dialog(folder,labels,entries,plot_sets[0],plot_sets[1]))
        open_button.grid(column=3,row=0)

        open_button1 = ttk.Button(root, text="Recalculate 456 Scan", command= lambda: recalculate456T(folder,labels,entries,plot_sets[0]))
        open_button1.grid(column=0,row=1)

        open_button2 = ttk.Button(root, text="Recalculate 456 Beat", command= lambda: recalculate456beat(folder,labels,entries,plot_sets[0]))
        open_button2.grid(column=4,row=1)

        open_button3 = ttk.Button(root, text="Recalculate 894 Scan", command= lambda: recalculate894T(folder,labels,entries,plot_sets[1]))
        open_button3.grid(column=0,row=5)

        open_button4 = ttk.Button(root, text="Recalculate 894 Beat", command= lambda: recalculate894beat(folder,labels,entries,plot_sets[1]))
        open_button4.grid(column=4,row=5)

        open_button4 = ttk.Button(root, text="Calculate 456 Abs Fit", command= lambda: calculate_abs_fit_456(folder,labels,plot_sets[0]))
        open_button4.grid(column=3,row=1)

        open_button4 = ttk.Button(root, text="Calculate 894 Abs Fit", command= lambda: calculate_abs_fit_894(folder,labels,plot_sets[1]))
        open_button4.grid(column=3,row=5)
        
        open_button5 = ttk.Button(root, text="Close", command= exit)
        open_button5.grid(column=3,row=9)

        open_button6 = ttk.Button(root, text="Record T Fits", command= lambda: record_fits(folder))
        open_button6.grid(column=3,row=4)
        
        

        workingdir_txt = tk.StringVar()
        workingdir_txt.set('hello')
        workingdir = ttk.Label(root, textvariable=workingdir_txt)
        workingdir.grid(column=4, row=8,columnspan=3, sticky="nsew")
    print(folder.folderpath)
    root.mainloop()
	