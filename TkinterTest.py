import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk

def open_file_dialog():
    filepath = filedialog.askdirectory(
        initialdir="/",  # Optional: set initial directory
        title="Select a folder",
        # filetypes=(("Text files", "*.txt"), ("All files", "*.*")) # Optional: filter file types
    )
    if filepath:
        print(f"Selected file: {filepath}")

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Geometry Calculation")
    open_button = ttk.Button(root, text="Open File", command=open_file_dialog)
    open_button.pack(pady=20)
    root.mainloop()
