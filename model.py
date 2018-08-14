import csv
import cv2



#read in the csv file containing the paths to the images and vehicle control data
lines = []
with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)


images = []
measurements = []

for line in lines:
	#get the center, left and right camera images
	for i in range(3):
		source_path = line[i]
		tokens = source_path.split('\\')
		filename = tokens[-1]
		local_path = "./data/IMG/" + filename
		image = cv2.imread(local_path)
		images.append(image)
	#get the steering angle for the center camera, add a correction value of 0.15 to the right and left cameras. 
	#If the car is going right, steer left. if the car is going left, steer right.
	correction = 0.15
	measurement = float(line[3])
	measurements.append(measurement)
	measurements.append(measurement+correction)
	measurements.append(measurement-correction)


augmented_images = []
augmented_measurements = []

#augmenting the dataset by flipping the images around vertical axis, add the corresponding inverse measurement
for image, measurement in  zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	flipped_image = cv2.flip(image, 1)
	flipped_measurement = float(measurement) * -1.0
	augmented_images.append(flipped_image)
	augmented_measurements.append(flipped_measurement)


import numpy as np

#create the training data as numpy arrays from the augmented dataset
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

import keras

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

#this model is based on Nvidia's End to End Deep Learning for Self-Driving Cars Network Architecture
model = Sequential()
#normalize the images by making the pixel values between -0.5 and 0.5
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
#crop the images to reduce process time, and also remove unnecessary information that may confuse the model. Rmove 70 pixels from top, 25 pixels from bottom. Leave sides untouched
model.add(Cropping2D(cropping=((70,25),(0,0))))
#5 Convolutional layers. The first 3 have a 5x5 kernel. There is a subsumpling layer of 2x2. This helps prevent overfitting.
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
#The last 2 conv layers have a kernel of 3x3, and no subsampling
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
#There are 4 Fully connected layers, the final layer being the output to drive the car
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
#adam optimizer used. Learning rate does not need to be tweaked. mse loss function used.
model.compile(optimizer='adam', loss='mse')
#validation split is 20%. 5 Epochs is used.
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')




