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

        

def open_file_dialog(test):
    temporary = filedialog.askdirectory(
        initialdir="/",  # Optional: set initial directory
        title="Select a folder",
        # filetypes=(("Text files", "*.txt"), ("All files", "*.*")) # Optional: filter file types
    )
    test.folderpath = temporary
    if test.folderpath:
        print(f"Selected folder: {test.folderpath}")
        test.checkforanalysis()

def update_path_label(text):
    workingdir_txt.set(value=text)

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
        Tavg456 = ttk.Frame(notebook456T)
        Pavg456 = ttk.Frame(notebook456T)
        scaledTnFit456 = ttk.Frame(notebook456T)
        correctedT = ttk.Frame(notebook456T)
        notebook456T.add(Tavg456, text="Tavg456")
        notebook456T.add(Pavg456, text="Pavg456")
        notebook456T.add(scaledTnFit456, text="scaled T Fit456")
        notebook456T.add(correctedT, text="Corrected T")


        label_Tavg_456 = tk.Label(Tavg456, image=plots456.Tavg)
        label_Tavg_456.image = plots456.Tavg
        label_Tavg_456.pack()
        
        # label_Tavg_456 = tk.Label(notebook456T, image=plots456.Tavg)
        # label_Tavg_456.image = plots456.Tavg
        # label_Tavg_456.pack()
        # notebook456T.add(Tavg456, text="Tavg456")
        
        notebook456Beat = ttk.Notebook(root)
        notebook456Beat.grid(column=2,row=0)
        ogbeat = ttk.Frame(notebook456Beat)
        filteredBeat = ttk.Frame(notebook456Beat)
        fittedBeat = ttk.Frame(notebook456Beat)
        unscaled_resid = ttk.Frame(notebook456Beat)
        scaled_resid = ttk.Frame(notebook456Beat)
        notebook456Beat.add(ogbeat, text="ogbeat")
        notebook456Beat.add(filteredBeat, text="filteredBeat")
        notebook456Beat.add(fittedBeat, text="fittedBeat")
        notebook456Beat.add(unscaled_resid, text="unscaled_resid")
        notebook456Beat.add(scaled_resid, text="scaled_resid")
        open_button = ttk.Button(root, text="Open Folder", command= lambda: open_file_dialog(folder))
        open_button.grid(column=1,row=0)
        workingdir_txt = tk.StringVar()
        workingdir_txt.set('hello')
    workingdir = ttk.Label(root, textvariable=workingdir_txt)
    workingdir.grid(column=1, row=1)
    print(folder.folderpath)
    root.mainloop()
	