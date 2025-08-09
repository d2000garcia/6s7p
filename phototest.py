from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
def change_image(old,new):
    old.configure(image=new)
    old.image = new


# pil_image = Image.open(r"C:\Users\wolfw\Downloads\BeatnoteProcess7-321-25\456beattest.png")
pil_image1 = Image.open(r"D:\Diego\git\6s7p\456beattest.png")
pil_image2 = Image.open(r"D:\Diego\git\6s7p\894beattest.png")
pil_image3 = Image.open(r".\Picture_template.png")
# pil_image.rotate(45).show()
plot_w = 500
plot_h = 300
resized_image1 = pil_image1.resize((plot_w, plot_h), Image.LANCZOS)
resized_image2 = pil_image2.resize((plot_w, plot_h), Image.LANCZOS)
resized_image3 = pil_image3.resize((plot_w, plot_h), Image.LANCZOS)
if __name__ == '__main__':
    root = tk.Tk()
    tk_image1 = ImageTk.PhotoImage(resized_image1)
    tk_image2 = ImageTk.PhotoImage(resized_image2)
    tk_image3 = ImageTk.PhotoImage(resized_image3)
    label1 = tk.Label(root, text = 'hello', image=tk_image1, compound=tk.TOP)
    label2 = tk.Label(root, image=tk_image2)
    # label1.image = tk_image1
    open_button = ttk.Button(root, text="Open Folder", command= lambda: change_image(label1, tk_image3))

    # label1.pack()
    # label2.pack()
    label1.grid(column=0,row=0)
    label2.grid(column=2, row=0)
    open_button.grid(column=2, row=1)
    root.mainloop()