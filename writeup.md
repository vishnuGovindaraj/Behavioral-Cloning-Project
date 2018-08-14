# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/center_image.png "center image"
[image2]: ./images/left_image.png "left image"
[image3]: ./images/right_image.png "right image"
[image4]: ./images/center_image_flipped.png "center image flipped"
[image5]: ./images/left_image_flipped_.png "left image cropped"
[image6]: ./images/right_image_flipped.png "right image cropped"
[image7]: ./images/center_image_flipped_cropped.png "center image flipped and cropped"
[image8]: ./images/left_image_flipped_cropped.png "left image flipped and cropped"
[image9]: ./images/right_image_flipped_cropped.png "right image flipped and cropped"
[image10]: ./images/nvidia-cnn-architecture.png "nvidia architecture"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

Note: I followed the Q&A provided by Udacity which can be found [here](https://www.youtube.com/watch?v=rpxZ87YFg0M&feature=youtube)

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network to run on track 1
* model2.h5 containing a trained convolution neural network to run on track 2
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the first track by executing 
```sh
python drive.py model.h5
```

The second track can be driven by executing

```sh
python drive.py model2.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I initially tried the LeNet5 Architecture which yielded poor results. After further research it was obvious that Nvidia's model is more powerful and suited specifically for this project.

This model is designed from Nvidia's network architecture for End to End deep learning for Self-Driving Cars.

Nvidia's model consists of 5 convolution layers (size 24-64), 3 of which have subsampling and 4 fully connected layers (size 100 - 1).(model.py lines 68-74) 

The model includes RELU layers to introduce nonlinearity (lines 68-74)

The data is normalized in the model using a Keras lambda layer (code line 64)

The data is cropped by a keras layer (code line 66)

#### 2. Attempts to reduce overfitting in the model

I initially tried dropout layers, which I had success with in the Traffic Sign Classifier project. I tried adding dropout in various areas: between and after conv layers, between and after fully connected layers. However this did not yield good results.
I increased the number of Epochs to compensate for the Dropout, but the validation and training loss did not end up being lower than before. The final performance run was also poorer so I decided to not use dropout.
I think that my dataset is not large enough to benefit from dropout.

I tried adding the second track to the dataset to generalize the data, but this resulted in poor results so my data is only from track1.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 81).

#### 4. Appropriate training data

I used 4 laps of training data to train the model. More details are in the following section

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I wanted to not use a python generator and my goal was to build a successful model with the least amount of data possible.

To capture good driving behavior, I recorded 4 laps in total. 2 of which were clockwise. 2 of which are anti clockwise. I found the best way of getting good simulator performance was to actually drive well. I found bad simulator performance was produced by driving poorly(near the edges) and too quickly. So slow and steady is a good way to go. After 4 laps of center-lane driving at a moderate speed(15mph), my model safely completes the track.

My model architecture is based on nvidia's end to end learning self-driving car deep learning network. This is a powerful network but it is only as good as the data it is provided and I noticed that the correlation between the quality of the data and the simulator performance is very high.

I did not find a need for recovery driving data.

I combatted overfitting by not running too many Epochs. After 5 Epochs the validation loss does not decrease which is a sign that the model is overfitting, so it's important to stop training here.

20% split for validation data is a standard size and it worked well in predicting real-time performance

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 64-79) consisted of a convolution neural network with the following layers and layer sizes:

Image input: 160x320x3, pixels normalized between (-0.5 , 0.5)
After Crop: 65x320x3

Convolution Layer 1: input depth 24, kernel 5x5, subsample layer (2x2) with relu activation
Convolution Layer 2: input depth 36, kernel 5x5, subsample layer (2x2) with relu activation
Convolution Layer 3: input depth 48, kernel 5x5, subsample layer (2x2) with relu activation
Convolution Layer 4: input depth 64, kernel 3x3 with relu activation
Convolution Layer 5: input depth 64, kernel 3x3 with relu activation

FlattenLayer

Fully Connected Layer 1: output depth 100
Fully Connected Layer 2: output depth 50
Fully Connected Layer 3: output depth 10
Fully Connected Layer 4: output depth 1

Here is a visualization of the architecture:

![alt text][image10]

#### 3. Creation of the Training Set & Training Process

To augment the dataset I used the left and right camera images, and a corrected steering angle (+0.15, -0.15). This increases the dataset by a factor of 3

![alt text][image1]Center Camera

![alt text][image2]Left camera

![alt text][image3]Right camera


To further augment the dataset I flipped all the images which increases the dataset by a factor of 2. So finally I get 6x the data from the original dataset.

![alt text][image4]
Center camera flipped
![alt text][image5]
Left camera flipped
![alt text][image6]
Right camera flipped

To reduce the process time and prevent confusing the model. I cropped the top 70 pixels (which removes a lot of the sky). The bottom 25 pixels (removes the hood of the car) from all the images. This was done by keras.

![alt text][image7]
Center camera flipped and cropped
![alt text][image8]
Left camera flipped and cropped
![alt text][image9]
Right camera flipped and cropped




