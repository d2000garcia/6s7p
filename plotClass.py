from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
class plots:
    def __init__(self,default_path = r".\Picture_template.png", plot_w = 500, plot_h = 300):
        default_img = Image.open(default_path)
        resized_default = default_img.resize((plot_w, plot_h), Image.LANCZOS)
        self.plot_w = plot_w
        self.plot_h = plot_h
        self.Tavg = resized_default.copy()
        self.Pavg = resized_default.copy()
        self.scaledT_and_fit = resized_default.copy()
        self.correctedT = resized_default.copy()
        self.ogbeat = resized_default.copy()
        self.filteredbeat = resized_default.copy()
        self.fitted_beat = resized_default.copy()
        self.unscaledresiduals = resized_default.copy()
        self.ScaledResiduals = resized_default.copy()
    
    def create_image(self):
        self.imagetk = ImageTk.PhotoImage(self.Tavg)
        self.imagetk2 = ImageTk.PhotoImage(self.Pavg)
    
    def update_image(self, whichone = '', plot_path ='',):
        temp = Image.open(plot_path)
        resized_temp = temp.resize((self.plot_w, self.plot_h), Image.LANCZOS)
        if whichone == 'Tavg':
            self.Tavg = resized_temp.copy()
        elif whichone == 'Pavg':
            self.Pavg = resized_temp.copy()
            self.imagetk2 = ImageTk.PhotoImage(self.Pavg)
        elif whichone == 'scaledT_and_fit':
            self.scaledT_and_fit = resized_temp.copy()
        elif whichone == 'correctedT':
            self.correctedT = resized_temp.copy()
        elif whichone == 'ogbeat':
            self.ogbeat = resized_temp.copy()
        elif whichone == 'filteredbea':
            self.filteredbeat = resized_temp.copy()
        elif whichone == 'fitted_beat':
            self.fitted_beat = resized_temp.copy()
        elif whichone == 'unscaledresiduals':
            self.unscaledresiduals = resized_temp.copy()
        elif whichone == 'ScaledResiduals':
            self.ScaledResiduals = resized_temp.copy()

def change_image(old,new):
    old.configure(image=new)
    old.image = new


# pil_image = Image.open(r"C:\Users\wolfw\Downloads\BeatnoteProcess7-321-25\456beattest.png")
# pil_image1 = Image.open(r"D:\Diego\git\6s7p\456beattest.png")
# pil_image2 = Image.open(r"D:\Diego\git\6s7p\894beattest.png")
# pil_image3 = Image.open(r".\Picture_template.png")
# pil_image.rotate(45).show()
# plot_w = 500
# plot_h = 300
# resized_image1 = pil_image1.resize((plot_w, plot_h), Image.LANCZOS)
# resized_image2 = pil_image2.resize((plot_w, plot_h), Image.LANCZOS)
# resized_image3 = pil_image3.resize((plot_w, plot_h), Image.LANCZOS)
first = True
temp = plots()
if __name__ == '__main__':
    root = tk.Tk()
    if first:
        temp.create_image()
        first = False
    notebook = ttk.Notebook(root)
    notebook.pack(expand=True, fill="both")
    tab1_frame = ttk.Frame(notebook)
    tab2_frame = ttk.Frame(notebook)
    notebook.add(tab1_frame, text="Tab 1")
    notebook.add(tab2_frame, text="Tab 2")
    # tk_image1 = ImageTk.PhotoImage(resized_image1)
    # tk_image2 = ImageTk.PhotoImage(resized_image2)
    # tk_image3 = ImageTk.PhotoImage(resized_image3)
    temp.update_image('Pavg',r"D:\Diego\git\6s7p\456beattest.png")
    label1 = tk.Label(tab1_frame, text = 'hello', image=temp.imagetk, compound=tk.TOP)
    label2 = tk.Label(tab1_frame, image=temp.imagetk)
    # label1.image = tk_image1
    open_button = ttk.Button(tab1_frame, text="Open Folder", command= lambda: change_image(label1, temp.imagetk2))

    # label1.pack()
    # label2.pack()
    label1.grid(column=0,row=0)
    label2.grid(column=2, row=0)
    open_button.grid(column=2, row=1)
    root.mainloop()