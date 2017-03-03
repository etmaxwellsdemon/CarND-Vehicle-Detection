##Vehicle Detection
###This is the project of Udacity Self-Driving Car

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
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

Before I finally decided to only use HOG features, I also did experiments over Color features. As it looks like that using HOG and Color features together would double the size of the feature and training/detection time. So I finally decided to only train HOG features on my computer.

The Color features is in the **Color Features** section in the **Vehicle Detection.ipynb**

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I took 6 examples of **vehicle** images and 6 for **non-vehicle** images. The code is in the **HOG Features** of the **Vehicle Detection.ipynb**

Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces. 

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters, and try to train the classifier to see the result.

For YCrCb, I got the accuracy 0.9825 by SVC, and only 0.9794 by YUV, other color spaces are even lower.

I've also tried to use features only in Y channel, and the accuracy for testing data still cannot catch up with YCrCb, so I decided to use **YCrCb** as the color space for HOG parameters.

For other parameters as `orientations`, `pixels_per_cell`, `cells_per_block`, I've tried to change them a little bit, like the `orientations` to 8, `pixels_per_cell=(8, 8)`, etc, the result didn't seems to be remarkablely improved. So I just decide to use the default parameter as in the class:

`orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM, based on YCrCb features, which is coded in the **Hog Classifier** section of the **Vehicle Detection.ipynb**. The size of the testing data is 20% of the whole data set. 

Finally, I've got the result of 0.9825 for YCrCb color space.

It can still be improved with more featues as bin spatial and color histogram, but will take more time for training and classify.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search window positions in the area that between 400px to 656px of y axis, this is because the car will always appear in the lower part of the image.

Also, I decide to search the scale at `1.5, 2, 3`, as 64x64 is too small for cars that display in the frame, and 1.5x will give us a 96x96 window, which sufficient for far-away cars. Further more, 1.5x window is 1.5 times faster than 1x window.

I also search 2x and 3x for near cars

The overlap is 2 cells per step, as 1 cell/step is too dense and slow, 2 is quite good for the next step.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.

From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. The threshould was set to **2**. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Then I combined the filtered heat for last 8 frames, set the threshould to **8** and identify individual blobls in the heatmap by the `scipy.ndimage.measurements.label()` method. The example of filtered result for last 8 frames is like this:


### Here are 8 frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all 8 frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?


Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

