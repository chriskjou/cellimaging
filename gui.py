from tkinter import filedialog
from tkinter import *
import boundaries

def callback():
    print("running cell counter...")
    print("Image Folder: " + folder_path.get())
    print("CSV Folder: " + csv_folder_path.get())
    endfolder = folder_path.get().split("/")[-1] + "/*"
    csvfolder = csv_folder_path.get().split("/")[-1] + "/*"
    boundaries.everything(endfolder, csvfolder)
    root.destroy()

def browse_button():
    global folder_path
    filename = filedialog.askdirectory()
    folder_path.set(filename)
    print(filename)

def csvbrowse_button():
    global csv_folder_path
    csvfilename = filedialog.askdirectory()
    csv_folder_path.set(csvfilename)
    print(csvfilename)

root = Tk()

folder_path = StringVar()
images = Label(text="Select Image Folder",font='Helvetica 13 bold')
images.grid(row=0, column=0)
lbl1 = Label(master=root,textvariable=folder_path)
lbl1.grid(row=0, column=1)
button2 = Button(text="Browse", command=browse_button)
button2.grid(row=0, column=3)

csv_folder_path = StringVar()
csvs = Label(text="Select CSV Folder", font='Helvetica 13 bold')
csvs.grid(row=1, column=0)
lbl2 = Label(master=root,textvariable=csv_folder_path)
lbl2.grid(row=1, column=1)
button3 = Button(text="Browse", command=csvbrowse_button)
button3.grid(row=1, column=3)

submit = Button(text="Submit", width=10, command=callback)
submit.grid(row=2, column=3)

mainloop()
