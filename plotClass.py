from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
class plots:
    def __init__(self,default_path = r".\Picture_template.png", scan = '456', plot_w = 500, plot_h = 300):
        default_img = Image.open(default_path)
        resized_default = default_img.resize((plot_w, plot_h), Image.LANCZOS)
        self.par_fold = ''
        self.scan = scan
        self.plot_w = plot_w
        self.plot_h = plot_h
        self.plotslabs=['Tavg','Pavg','scaledT','FittedScan','FittedScanResid','ogbeat','filteredbeat','fitted_beat','unscaledresiduals','Temperature']
        self.plots = []
        for i in self.plotslabs:
            self.plots.append(ImageTk.PhotoImage(resized_default.copy()))
        if scan == '456':
            self.fold = self.par_fold + r'\Analysis\456\plots'
        else:
            self.fold = self.par_fold + r'\Analysis\894\plots'
    
    def update_working_dir(self, new_par_fold):
        self.par_fold = new_par_fold
        if self.scan == '456':
            self.fold = self.par_fold + r'\Analysis\456\plots'
        else:
            self.fold = self.par_fold + r'\Analysis\894\plots'

    def update_image(self, tochange):
        for whichone in tochange:
            plot_path = self.fold + '\\' + whichone + '.png'
            temp = Image.open(plot_path)
            resized_temp = temp.resize((self.plot_w, self.plot_h), Image.LANCZOS)
            if whichone == 'Tavg':
                self.plots[0] = ImageTk.PhotoImage(resized_temp)
            elif whichone == 'Pavg':
                self.plots[1] = ImageTk.PhotoImage(resized_temp)
            elif whichone == 'scaledT':
                self.plots[2] = ImageTk.PhotoImage(resized_temp)
            elif whichone == 'FittedScan':
                self.plots[3] = ImageTk.PhotoImage(resized_temp)
            elif whichone == 'FittedScanResid':
                self.plots[4] = ImageTk.PhotoImage(resized_temp)
            elif whichone == 'ogbeat':
                self.plots[5] = ImageTk.PhotoImage(resized_temp)
            elif whichone == 'filteredbeat':
                self.plots[6] = ImageTk.PhotoImage(resized_temp)
            elif whichone == 'fitted_beat':
                self.plots[7] = ImageTk.PhotoImage(resized_temp)
            elif whichone == 'unscaledresiduals':
                self.plots[8] = ImageTk.PhotoImage(resized_temp)
            elif whichone == 'Temperature':
                self.plots[9] = ImageTk.PhotoImage(resized_temp)

def change_Label_image(oldlabel,new):
    #oldlabel is the label you want to change and
    #new is new Tkimage to exchange
    oldlabel.configure(image=new)
    oldlabel.image = new


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
# first = True
# temp = plots()
# if __name__ == '__main__':
#     root = tk.Tk()
#     if first:
#         temp.create_image()
#         first = False
#     notebook = ttk.Notebook(root)
#     notebook.pack(expand=True, fill="both")
#     tab1_frame = ttk.Frame(notebook)
#     tab2_frame = ttk.Frame(notebook)
#     notebook.add(tab1_frame, text="Tab 1")
#     notebook.add(tab2_frame, text="Tab 2")
#     # tk_image1 = ImageTk.PhotoImage(resized_image1)
#     # tk_image2 = ImageTk.PhotoImage(resized_image2)
#     # tk_image3 = ImageTk.PhotoImage(resized_image3)
#     temp.update_image('Pavg',r"D:\Diego\git\6s7p\456beattest.png")
#     label1 = tk.Label(tab1_frame, text = 'hello', image=temp.imagetk, compound=tk.TOP)
#     label2 = tk.Label(tab1_frame, image=temp.imagetk)
#     # label1.image = tk_image1
#     open_button = ttk.Button(tab1_frame, text="Open Folder", command= lambda: change_image(label1, temp.imagetk2))

#     # label1.pack()
#     # label2.pack()
#     label1.grid(column=0,row=0)
#     label2.grid(column=2, row=0)
#     open_button.grid(column=2, row=1)
#     root.mainloop()