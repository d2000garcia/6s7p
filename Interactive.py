import os as os
import numpy as np
from numpy.polynomial import Polynomial as poly
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk

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
                os.mkdir(self.folderpath+r'\Analysis\894')
                os.mkdir(self.folderpath+r'\Analysis\894\beatnote')
                os.mkdir(self.folderpath+r'\Analysis\894\beatnote\original')
                os.mkdir(self.folderpath+r'\Analysis\894\beatnote\processed')
                os.mkdir(self.folderpath+r'\Analysis\894\fitting')
                os.mkdir(self.folderpath+r'\Analysis\894\fitting\original')
                os.mkdir(self.folderpath+r'\Analysis\894\fitting\processed')
                os.mkdir(self.folderpath+r'\Analysis\894\plots')
            # self.folderpath_tkvar.set(self.folderpath)
            update_path_label(self.folderpath)

        

def open_file_dialog(analysis_dat,labels,plots456,plots894, switchlabelsat=4):
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
            plots456.update_image(lab)
            plots894.update_image(lab)
            if i < 5:
                change_Label_image(labels[0], plots456.plots[i])
                change_Label_image(labels[2], plots894.plots[i])
            else:
                change_Label_image(labels[1], plots456.plots[i])
                change_Label_image(labels[3], plots894.plots[i])
        

            


def update_path_label(text):
    workingdir_txt.set(value=text)

def change_Label_image(oldlabel,new):
    #oldlabel is the label you want to change and
    #new is new Tkimage to exchange
    oldlabel.configure(image=new)
    oldlabel.image = new

first = True
template_image = r".\Picture_template.png"
if __name__ == '__main__':
    root = tk.Tk()
    if first:
        first = False
        folder = analysis()
        plots456 = plots(scan='456')
        plots894 = plots(scan='894')
        root.title("Geometry Calculation")

        notebook456T = ttk.Notebook(root)
        notebook456T.grid(column=0,row=0)
        labels = [[],[],[],[]]
        #labels = [labels456T,labels456Beat, labels894T,labels894Beat]
        
        for i, lab in enumerate(plots456.plotslabs[:4]):
            labels[0].append(tk.Label(notebook456T,image=plots456.plots[i]))
            labels[0][-1].image = plots456.plots[i]
            labels[0][-1].pack()
            notebook456T.add(labels[0][-1],text=lab)

        notebook456Beat = ttk.Notebook(root)
        notebook456Beat.grid(column=2,row=0)
        for i, lab in enumerate(plots456.plotslabs[4:],start=4):
            labels[1].append(tk.Label(notebook456Beat,image=plots456.plots[i]))
            labels[1][-1].image = plots456.plots[i]
            labels[1][-1].pack()
            notebook456Beat.add(labels[1][-1],text=lab)
        

        notebook894T = ttk.Notebook(root)
        notebook894T.grid(column=0,row=2)
        for i, lab in enumerate(plots894.plotslabs[:4]):
            labels[2].append(tk.Label(notebook894T,image=plots894.plots[i]))
            labels[2][-1].image = plots894.plots[i]
            labels[2][-1].pack()
            notebook894T.add(labels[2][-1],text=lab)

        notebook894Beat = ttk.Notebook(root)
        notebook894Beat.grid(column=2,row=2)
        for i, lab in enumerate(plots894.plotslabs[4:],start=4):
            labels[3].append(tk.Label(notebook894Beat,image=plots894.plots[i]))
            labels[3][-1].image = plots894.plots[i]
            labels[3][-1].pack()
            notebook894Beat.add(labels[3][-1],text=lab)

        open_button = ttk.Button(root, text="Open Folder", command= lambda: open_file_dialog(folder,labels,plots456,plots894))
        open_button.grid(column=1,row=0)
        workingdir_txt = tk.StringVar()
        workingdir_txt.set('hello')
    workingdir = ttk.Label(root, textvariable=workingdir_txt)
    workingdir.grid(column=1, row=1)
    print(folder.folderpath)
    root.mainloop()
	