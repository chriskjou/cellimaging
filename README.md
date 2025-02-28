# cellimaging

## Installation

Make sure you have [Python](https://www.python.org/) installed, then install [Tensorflow](https://www.tensorflow.org/install/) on system, then clone this repo.

Image classification with transfer learning from Google's Inveption v3 already done with models and labels uploaded in Dropbox links for [brightfield](https://www.dropbox.com/s/5ulmpibmutedal5/logs-brightfield.zip?dl=0) and [fluorescent](https://www.dropbox.com/s/ezrkoq98ucwd2mu/logs-fluorescence.zip?dl=0). Both folders must be unzipped and placed directly in the cellimaging folder as indicated below.

```
|
---- /cellimaging
|    |
|    |
|    ---- /logs-brightfield
|    |    trained_graph.pb
|    |    trained_labels.txt
|    |
|    ---- /logs-fluorescence
|    |    trained_graph.pb
|    |    trained_labels.txt
|    |
|    ---- /images
|    |    firstimagec1t1.tiff
|    |    firstimagec12t1.tiff
|    |    secondimagec1t2.tiff
|    |    secondimagec12t2.tiff
|         ...
|
```

Ensure that requirements are installed by running
```
pip install -r requirements.txt
```

Note that logs-brightfield and logs-fluoresecence must maintain the same folder name. The image folder you are analyzing is not restricted by naming.

## Usage
```
python gui.py
```

Select image folder and click submit. Progress is documented in terminal.


## File Descriptions
gui.py: runs user input for image folder with GUI

- for each timepoint/location, must have two corresponding images (brightfield and fluoresecent)
    
- naming of files must end with c1t* for brightfield and c2t* for fluorescent, otherwise same
    
- note: folder of images MUST be within the 'cellimaging' folder directory

### BEHIND THE SCENES:

#### getting cell centers (run either getCounts.txt for manual OR getcenters.py)
getCounts.txt: macro for ImageJ to get coordinates of selected points (for centers of droplets)

getcenters.py: automated cell selection and numbering with Canny edge detection and modified Hough Circle transform with thresholded area and circularity

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
