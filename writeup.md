## Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/00_car_not_car.png
[image2]: ./output_images/01_car_not_car_hog.png
[image3]: ./output_images/03_window.png
[image4]: ./output_images/02_result.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the lines through 12 to 37 of `save_svc.py`. The actual working code is in `get_hog_features()` functions (lines #14 through #31 and #51 through #99 of the file called `lesson_functions.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images. The code for this is contained in the `get_image_list()` function of `utils.py`

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried several combinations of colorspaces.
Here are the test accuracies varied with colorspaces.

| Color Space | Test Accuracy of SVC |
| --- | --- |
| RGB | 0.9797 |
| HSV | 0.9918 |
| YCrCb | 0.993 |

I've chose to follow setup from lessons, which is Colorspace=YCrCb, orient=9, pix_per_cell=8, cell_per_block=2. Because it shows quite good results by itself and it doesn't seem to get better significantly with changing parameters.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this step is contained in the lines through 39 to 79 of `save_svc.py` file. 
I've trained SVC with spatial_features, hist_features, and hog_features.

Here's how I did.
1. I've extracted Features, (hog, spatial, and hist)
1. Normalized with `StandardScaler`
1. Randomize & split set into train & test set with `train_test_split` 
1. Train with `LinearSVC`
1. Save to trained svc to `svc_pickle.p`

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for this step is contained in the `find_cars()` function of `lesson_functions.py` file. 
~~I have followed exact setup used in lessons.~~
~~The scale is 1.5. Pix per cell is 8 and cell per block is 2, it means 75% of overlaps.~~

**Following the suggestions from review, I used multiple scale (1.0, 1.5, and 2.0).
I've also reduced cell per block to 1, so the overlap is now 87.5%.**

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image3]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, and the bounding boxes then overlaid on the example images:

**Following the suggestion from review, I have used `collections.deque` to accumulate heatmap.**

### Here are six frames and their corresponding heatmaps, column 3, and the resulting bounding boxes are drawn onto the last frame in the series, column 4: 
![alt text][image4]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The result can be more smooth if I track the result of previous several frames and decide if its valid or not.
The speed of pipeline seems acceptable, 5 fps in my laptop (2013 rMBP 15"). 

