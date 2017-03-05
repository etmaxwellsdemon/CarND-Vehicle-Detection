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
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/sliding_window1.png
[image4]: ./output_images/sliding_window2.png
[image5]: ./output_images/sliding_window3.png
[image6]: ./output_images/sliding_window4.png
[image7]: ./output_images/sliding_window5.png
[image8]: ./output_images/t1_10|12_17|18.png
[video1]: ./output_images/output_10|12_17|18.mp4

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

I decided to search window positions in the area that between 380px to 656px of y axis, this is because the car will always appear in the lower part of the image.

Also, I tried to search the scale at `1, 1.5, 2, 3`, and finally decided to only use `1.5x and 3x` for scale parameter, as 1.5x will give us a 96x96 window, which sufficient for far-away cars. Further more, 1.5x window is 1.5 times faster than 1x window.

The overlap is 2 cells per step, as 1 cell/step is too dense and slow, 2 is quite good for the next step.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)
Also in [YouTube](https://youtu.be/Sb87V5ln7G0)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

For pipeline, I decide to filter the output.

1. I output the heat by each frame, then I combined heat for last `m` frames, set threashould and output them as `heat_combine`, This is so called `filter1`
2. For output of `heat_combine`, I set all positive pixals to 1, this can decrease the effect of false positives.
3. Then I sum `n` previous `heat_combine`, add threashould and output them as `heat2_combine`, This is so called `filter2` 

Then I tuned the parameter of `m`,`n` and threshould for each step.
I recorded the positions of positive detections in each frame of the video.

Finally I got the filter parameter as bellow:

For the output from each frame, I set threshould to 0, this will increase number of positive detections from ignoring some correct but hard to detect frames.

For the `filter1`, I set `m` to 12 and threshould to 10.

For the `filter2`, I set `n` to 18 and threshould to 17.

I've selected outputs of several frames in the test video:

![alt text][image8]

* The 1st column is the orignal output for window searching.
* The 2nd column is its related heatmap.
* The 3rd column is the result after 1st filter
* The 4th column is the result after 2nd filter
* The 5th column is the final output


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

1. I only use HOG feature to classify, this is faster than use both HOG and color features, but the accuary is only 98.25%, which can generate lots of false positive results. The classifer can be improved with more features.
2. I only use LinearSVM classifier, and I haven't tried other classifiers as deep neuro networks, which may generate better result.
3. As there're lots of false positive detections, and the classifier will fail to detect car in certain frames, this make the final pipeline harder to filter. That's why I used two average filter connect by each other to generate a better result. 
4. Although the filtered result is much better, the filter can still be improved based on other informations, such as:
   * The trajectory of the car
   * The possible position of the car
5. Also, for cons, a filter with large window may also slow down the reaction rate for real-time unman vehicles.

