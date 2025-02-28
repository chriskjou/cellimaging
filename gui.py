from tkinter import filedialog
from tkinter import *
import boundaries
import pandas as pd
from datetime import datetime
import time
import os

def callback():
    print("Image Folder: " + folder_path.get())
    print("\nrunning...\n")
    t0 = time.time()
    folder = folder_path.get().split("/")[-1] + "/*"
    imageids, cellnums, infected, cells = boundaries.everything(folder)
    returncsv(imageids, cellnums, infected, cells)
    t1 = time.time()
    print("total time: " + str(round(t1-t0)) + " seconds")
    root.destroy()
    sys.exit()

def returncsv(imageids, cellnums, infected, cellcount):
    sub = pd.DataFrame()
    sub['ImageId'] = imageids
    sub['Cell #'] = cellnums
    sub['Total'] = cellcount
    sub['Infected'] = infected
    sub['Viability (%)'] = (1 - sub['Infected']/sub['Total']) * 100
    directory = "outputcsvs"
    if not os.path.exists(directory):
        os.makedirs(directory)
    csvname = "outputcsvs/" + datetime.now().strftime('%Y-%m-%d=%H-%M-%S') + '.csv'
    sub.to_csv(csvname, index=False)
    print("\nfinished exporting to " + csvname + "...")

def browse_button():
    global folder_path
    filename = filedialog.askdirectory()
    folder_path.set(filename)
    print(filename)

root = Tk()
root.title('cell counter')
folder_path = StringVar()
images = Label(text="Select Image Folder",font='Helvetica 13 bold')
images.grid(row=0, column=0)
lbl1 = Label(master=root,textvariable=folder_path)
lbl1.grid(row=0, column=1)
button2 = Button(text="Browse", command=browse_button)
button2.grid(row=0, column=3)

submit = Button(text="Submit", width=10, command=callback)
submit.grid(row=1, column=3)

mainloop()
