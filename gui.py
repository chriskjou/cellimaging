from tkinter import filedialog
from tkinter import *
import boundaries
import find
import pandas as pd
from datetime import datetime
import time

def callback():
    print("running cell counter...")
    print("Image Folder: " + folder_path.get())
    print("CSV Folder: " + infected_folder_path.get())
    print("CSV Folder: " + csv_folder_path.get())
    t0 = time.time()
    endfolder = folder_path.get().split("/")[-1] + "/*"
    ogfolder = folder_path.get().split("/")[-1] + "/*"
    csvfolder = csv_folder_path.get().split("/")[-1] + "/*"
    imageids, cellnums, infected = boundaries.everything(endfolder, csvfolder)
    cellcount = find.foreachfile(ogfolder, csvfolder)
    returncsv(imageids, cellnums, infected, cellcount)
    t1 = time.time()
    print("total time: " + str(round(t1-t0)))
    root.destroy()

def returncsv(imageids, cellnums, infected, cellcount):
    sub = pd.DataFrame()
    sub['ImageId'] = imageids
    sub['Cell #'] = cellnums
    sub['Total'] = cellcount
    sub['Infected'] = infected
    sub['Viability'] = 1 - sub['Infected']/sub['Total']
    csvname = datetime.now().strftime('%Y-%m-%d=%H-%M-%S') + '.csv'
    sub.to_csv(csvname, index=False)
    print("finished exporting to " + csvname + "...")

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

def infectedbrowse_button():
    global infected_folder_path
    infectedfilename = filedialog.askdirectory()
    infected_folder_path.set(infectedfilename)
    print(infectedfilename)

root = Tk()

folder_path = StringVar()
images = Label(text="Select Image Folder",font='Helvetica 13 bold')
images.grid(row=0, column=0)
lbl1 = Label(master=root,textvariable=folder_path)
lbl1.grid(row=0, column=1)
button2 = Button(text="Browse", command=browse_button)
button2.grid(row=0, column=3)

infected_folder_path = StringVar()
infected = Label(text="Select Infected Folder", font='Helvetica 13 bold')
infected.grid(row=1, column=0)
lbl2 = Label(master=root,textvariable=infected_folder_path)
lbl2.grid(row=1, column=1)
button3 = Button(text="Browse", command=infectedbrowse_button)
button3.grid(row=1, column=3)

csv_folder_path = StringVar()
csvs = Label(text="Select CSV Folder", font='Helvetica 13 bold')
csvs.grid(row=2, column=0)
lbl3 = Label(master=root,textvariable=csv_folder_path)
lbl3.grid(row=2, column=1)
button4 = Button(text="Browse", command=csvbrowse_button)
button4.grid(row=2, column=3)

submit = Button(text="Submit", width=10, command=callback)
submit.grid(row=3, column=3)

mainloop()
