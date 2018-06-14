# cellimaging

### USER RUNS: python gui.py
gui.py: runs user input for image folder with GUI

- for each timepoint/location, must have two corresponding images (brightfield and fluoresecent)
    
- naming of files must end with c1t* for brightfield and c2t* for fluorescent, otherwise same
    
- note: folder of images MUST be within the 'cellimaging' folder directory

### BEHIND THE SCENES:

#### getting cell centers (run either getCounts.txt for manual OR getcenters.py)
getCounts.txt: macro for ImageJ to get coordinates of selected points (for centers of droplets)

getcenters.py: automated cell selection and numbering with Canny edge detection and Hough Circle transform with thresholded area and concavity

#### before machine learning
blobdetector.py: example of image detection with no machine learning

#### main code

boundaries-original.py: original code with keras model for wellplate (in progress)

boundaries.py: only with google inception v3 transfer learning

canny.py: detecting with Watershed and distance transform (in progress)

classify.py: identifies brightfield photos

classify2.py: identifies fluoresecent photos

convert.py: 90, 180, 270 rotation with reflection (vertical and horizontal) for initial manual labelling

helpers.py: required helper functions for boundaries

#### TBA: creating executable from only gui.py

pretrained models and labels from inception:
- brightfield: https://www.dropbox.com/s/5ulmpibmutedal5/logs-brightfield.zip?dl=0
- fluorescent: https://www.dropbox.com/s/ezrkoq98ucwd2mu/logs-fluorescence.zip?dl=0
