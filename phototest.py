from PIL import Image, ImageTk
import tkinter as tk

pil_image = Image.open(r"C:\Users\wolfw\Downloads\BeatnoteProcess7-321-25\456beattest.png")
pil_image.rotate(45).show()

# resized_image = pil_image.resize((width, height), Image.LANCZOS)
tk_image = ImageTk.PhotoImage(pil_image)

root = tk.Tk()
label = tk.Label(root, image=tk_image)
label.pack()
root.mainloop()