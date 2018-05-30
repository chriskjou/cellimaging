# cellimaging

getscreenshot.py: takes screenshots of subimages according to centers of droplets given in csv file

withsubfolder.py: uses getscreenshot.py to get subimages for each folder and count cells in each droplet (not accurate because of localization)

nosubfolder.py: route to desired folder with tiff/png images and returns cell count in csv

getCounts.txt: macro for ImageJ to get coordinates of selected points (for centers of droplets)

TBA: working on getting boundary of masks and retraining Google's Inceptionv3 Net for initial infection vs. no infection due to black amplifying noise
