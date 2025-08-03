import os as os
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk

class analysis:
    def __init__(self):
        self.folderpath = ''
    
    # def checkforanalysis(self):
    #     if not analysis.folderpath == '':
    #         contents = os.listdir(path = analysis.folderpath)
    #         if 'Analysis' in contents:
    #             print('Analysis exists, continue')
    #         else:
    #             print('Analysis does not exist ')
    #             os.mkdir(analysis.folderpath+r'\Analysis')
        

def open_file_dialog():
    test = filedialog.askdirectory(
        initialdir="/",  # Optional: set initial directory
        title="Select a folder",
        # filetypes=(("Text files", "*.txt"), ("All files", "*.*")) # Optional: filter file types
    )
    if test.folderpath:
        print(f"Selected file: {test}")

if __name__ == '__main__':
    root = tk.Tk()
    folder = analysis
    root.title("Geometry Calculation")
    open_button = ttk.Button(root, text="Open Folder", command=open_file_dialog)
    open_button.pack(pady=20)
    print(folder.folderpath)
    root.mainloop()
	