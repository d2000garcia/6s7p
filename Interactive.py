import os as os
import numpy as np
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk

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
        print(f"Selected file: {test.folderpath}")
        test.checkforanalysis()

def update_path_label(text):
    workingdir_txt.set(value=text)

if __name__ == '__main__':
    root = tk.Tk()
    folder = analysis()
    root.title("Geometry Calculation")
    open_button = ttk.Button(root, text="Open Folder", command= lambda: open_file_dialog(folder))
    open_button.pack(pady=20)
    workingdir_txt = tk.StringVar()
    workingdir_txt.set('hello')
    workingdir = ttk.Label(root, textvariable=workingdir_txt)
    workingdir.pack(pady=20)
    print(folder.folderpath)
    root.mainloop()
	