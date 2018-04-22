# Vehicle_Detection_Tracking
Udacity MOOC project

## Writeup 

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image2]: ./examples/hogfeats.jpg
[image4]: ./examples/Pipeline.jpg
[image3]: ./examples/colorspace_experimentation.jpg
[image5]: ./examples/heatmap.jpg
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README



### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

Section 1 in code.

I  explored different color spaces and different parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `get_hog_feats()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=11`, `pix_per_cell=16` and `cell_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and settled for a combination of speed and accuracy (see below). YUV and YCrCb achieved similar results but the latter required less scale adjustment.

    ### Parameter seclection
    color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb; NB if YUV, need to scale back up *255 +++
    orient = 11  # HOG orientations
    pix_per_cell = 16 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16) # Spatial binning dimensions
    hist_bins = 16    # Number of histogram bins
    spatial_feat = True # Spatial features on or off
    hist_feat = False # Histogram features on or off
    hog_feat = True # HOG features on or off
    
See also the colorspace_experimentation.ipynb notebook for insight into choice of colorspace (here's a teaser).
![alt text][image3]


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained both a linear SVM and an RBF-SVM. In the end I used the latter for the video because it generalized much better to the new images.

Importantly, I only scaled the data using information from the training set, to maintain blindness to test data (unlike in the supplied code, which scales before the train-test split).
```python
from sklearn.preprocessing import StandardScaler
```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Eyeballing the images shows that cars tend to take up pixel-spaces corresponding to the interval between scales 1 and 2. After experimenting with different numbers of scales, 3 was enough. Instead of overlap, my final solution uses cells_per_step=2.

![alt text][image5]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I trained both SVMs using parameters chosen through cross-validation (which took >790 seconds). See "Parameter selection" in the notebook.

Here are some random frames showing how the pipeline works (including imperfections).

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output/project_video_solution5.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Going from training-test data to the video resulted in lots of false positives. Initially I had used about 60% of available data and added more in through hard negative mining to decrease the false positive rate. This helped, but not as much as simply using all data from the beginning, which is what my final model uses. Going forward I would use yet more data, possibly through augmentation.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The classifier still identifies lots of false positives (and some crucial false negatives), which could probably be substantially corrected using classes to threshold detections by conviction (as in Advanced Lane Finding).

Like lots of these projects this process will likely fail going up and down hills, as the perspective used in training is always from the same height as the target.
