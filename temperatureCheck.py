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

def open_file_dialog(analysis_dat,labels,entries,plots456,plots894,entries2,switchlabelsat=5):
    temporary = filedialog.askdirectory(
        initialdir="/",  # Optional: set initial directory
        title="Select a folder",
        # filetypes=(("Text files", "*.txt"), ("All files", "*.*")) # Optional: filter file types
    )

Day_folder = 'test'

if __name__ == '__main__':
    scale = 1.2
    plot_w= int(500*scale)
    plot_h= int(300*scale)
    root = tk.Tk()
    if first:
        first = False
        folder = analysis()
        plot_sets= [plots(scan='456',plot_w=plot_w,plot_h=plot_h),plots(scan='894',plot_w=plot_w,plot_h=plot_h)]
        # plots456 = plots(scan='456')
        # plots894 = plots(scan='894')
        root.title("++FittingCalculations")

        notebooks= [ttk.Notebook(root),ttk.Notebook(root),ttk.Notebook(root),ttk.Notebook(root)]
        #notebooks = [456T,456Beat,894T,894Beat]
        notebooks[0].grid(column=0,row=0,columnspan=3, sticky="nsew")
        notebooks[1].grid(column=4,row=0,columnspan=3, sticky="nsew")
        notebooks[2].grid(column=0,row=4,columnspan=3, sticky="nsew")
        notebooks[3].grid(column=4,row=4,columnspan=3, sticky="nsew")


        entry_lbl = [ttk.Label(root, text='456  Background Linear Fit Ranges'), ttk.Label(root, text='456 Beatnote Range')]
        entry_lbl.append(ttk.Label(root, text='Upper-Baseline'))
        entry_lbl.append(ttk.Label(root, text='Lower-Baseline'))
        entry_lbl.append(ttk.Label(root, text='894 Background Linear Fit Rngs'))
        entry_lbl.append(ttk.Label(root, text='894 Beatnote Range'))
        entry_lbl.append(ttk.Label(root, text='Upper-Baseline'))
        entry_lbl.append(ttk.Label(root, text='Lower-Baseline'))
        entry_lbl.append(ttk.Label(root, text='Left Group'))
        entry_lbl.append(ttk.Label(root, text='Right Group'))
        entry_lbl.append(ttk.Label(root, text='456 fit etalon effect'))
        entry_lbl.append(ttk.Label(root, text='894 etalon etalon effect'))

        entry_lbl[0].grid(column=1,row=1,columnspan=2, sticky="nsew")
        entry_lbl[1].grid(column=5,row=1,columnspan=2, sticky="nsew")
        entry_lbl[2].grid(column=0,row=2)
        entry_lbl[3].grid(column=0,row=3)
        entry_lbl[4].grid(column=1,row=5,columnspan=2, sticky="nsew")
        entry_lbl[5].grid(column=5,row=5,columnspan=2, sticky="nsew")
        entry_lbl[6].grid(column=0,row=6)
        entry_lbl[7].grid(column=0,row=7)
        entry_lbl[8].grid(column=0,row=9)
        entry_lbl[9].grid(column=0,row=10)
        entry_lbl[10].grid(column=1,row=8,columnspan=2)
        entry_lbl[11].grid(column=5,row=8,columnspan=2)


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


        beat_mins = [[ttk.Entry(root,width=ent_wdth), ttk.Entry(root,width=ent_wdth)]]
        beat_mins.append([tk.StringVar(value="0"), tk.StringVar(value="0")])
        beat_mins[0][0].grid(column=5,row=3)
        beat_mins[0][1].grid(column=5,row=7)
        beat_mins[0][0].configure(textvariable=beat_mins[1][0])
        beat_mins[0][1].configure(textvariable=beat_mins[1][1])

        entry = ttk.Entry(root,width=ent_wdth)

        etalon_entries = [[[ttk.Entry(root,width=ent_wdth),ttk.Entry(root,width=ent_wdth),ttk.Entry(root,width=ent_wdth),ttk.Entry(root,width=ent_wdth)],[ttk.Entry(root,width=ent_wdth),ttk.Entry(root,width=ent_wdth),ttk.Entry(root,width=ent_wdth),ttk.Entry(root,width=ent_wdth)]]]
        etalon_entries.append([[tk.StringVar(value="0"),tk.StringVar(value="0"),tk.StringVar(value="0"),tk.StringVar(value="0")],[tk.StringVar(value="0"),tk.StringVar(value="0"),tk.StringVar(value="0"),tk.StringVar(value="0")]])
        for i in range(2):
            #i=0 is 456
            for j in range(4):
                etalon_entries[0][i][j].configure(textvariable=etalon_entries[1][i][j])
                # print(etalon_entries[1][i][j].get())
                # print(1+j%2+4*i,(j>1)+8)
                etalon_entries[0][i][j].grid(column=1+j%2+4*i,row=(j>1)+9)

        open_button = ttk.Button(root, text="Data Folder", command= lambda: open_file_dialog(folder,labels,entries,plot_sets[0],plot_sets[1],etalon_entries[1]))
        open_button.grid(column=3,row=0)

        open_button1 = ttk.Button(root, text="Calculate 456 baseline ratio", command= lambda: recalculate456T(folder,labels,entries,plot_sets[0]))
        open_button1.grid(column=2,row=1)
        
        open_button5 = ttk.Button(root, text="Close", command= exit)
        open_button5.grid(column=3,row=10)
        
        workingdir_txt = tk.StringVar()
        workingdir_txt.set('hello')
        workingdir = ttk.Label(root, textvariable=workingdir_txt)
        workingdir.grid(column=3, row=11,columnspan=3, sticky="nsew")
    print(folder.folderpath)
    root.mainloop()
	