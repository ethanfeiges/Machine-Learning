#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Using a sequential model for neural network
from keras.models import Sequential
# Tools for flattening an image. (Creates proper setup)
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# In[2]:


## Classifier neural network: Classifies between two things (yes/no)
## Code will determine whether the image is a dog or a cat
classifier = Sequential()

## Step 1/2 - Convolution. 
## Adding the first layer of the neural network: Your input
## input_shape: Your picture is coming in at 64 by 64 pixels with 3 values
## Activation (weighs the sum): Relu function deadends 0 to 1 values. 
## Conv2D: Converts photo to a two-dimensional setup.
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))


## HIDDEN LAYERS:
## Second convolutional layer:
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
## Mapping then reducing: Reducing the data to only 2 sets (64, 64, 3) to (2, 2). (No 3rd dimension of color)
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# In[3]:


## Step 3 - Flattening: Turning 2D array of pixels (single dimension) into a single array (64 * 64 2D -> 64^2 single array)
classifier.add(Flatten())

## Step 4 - Full connection. 
## Reducing the array to 128
classifier.add(Dense(units = 128, activation = 'relu'))

## Single output: (True/false, dog/cat)
## Sigmoid activiation either yields values of 1 or 0 (yes or no)
classifier.add(Dense(units = 1, activation = 'sigmoid'))


# In[4]:


## Compiling the classifier neural network:
## loss and metrics: How error is calculated
## optimizer: reverse propagation (adjusting weights during training)
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[5]:


## Part 2 - Fitting the Classifier neural network to the images (supervised training)
## 10,000 images total. 8,000 will be used for training, 2,000 will be used for testing
from keras.preprocessing.image import ImageDataGenerator
## If the photo is different shapes or sizes
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = .2, zoom_range = .2, horizontal_flip = True)

## batch_size: 32 images will be "batched" through the training each time.
## target_size: 
training_set = train_datagen.flow_from_directory(r"C:\Users\ejfei\Downloads\training_set\training_set", target_size = (64, 64), batch_size = 32, class_mode = 'binary')


# In[6]:


## formatting the test_data images
test_datagen = ImageDataGenerator(rescale = 1./255)

## Loading test set 
test_set = test_datagen.flow_from_directory(r"C:\Users\ejfei\Downloads\test_set\test_set", 
                                            target_size = (64, 64), 
                                            batch_size = 32, 
                                            class_mode = 'binary')


# In[31]:


## Back propogation to allow classifier to update weights by propagating how much of the loss each node is responsible for
## loss: Penalty for bad prediction. (Perfect loss is zero) FORMULA: [(actual output) - (predicted output)]^2
## Epoch: We will go through the whole data set 10 times
## steps_per_epoch: We will look at 200 pictures per each "epoch"/process
## validation_data: test_set will validate the training set and evaluate the loss
## validation-steps: 10 images from the test_set for each validation
## validation: data that has never been seen before by the classifier
## accuracy: The accuray of what the classifier is training
## val_accc: The accuracy of the testing data
## "fitting" your model (the more data "looked at" the more acccurate the model will be)
classifier.fit(training_set, steps_per_epoch = 200, 
                         epochs = 10, 
                         validation_data = test_set, 
                         validation_steps = 10)


# In[49]:


## Part 3 - Making new predictions
import numpy as np
from keras.preprocessing import image
## Loading testing image (completely new) that is a dog
test_image = image.load_img(r"C:\Users\ejfei\OneDrive\Pictures\corgi.jpg", target_size = (64, 64))
## 
test_image = image.img_to_array(test_image)
## converting over to a single array
test_image = np.expand_dims(test_image, axis = 0)
## Creates 2d array where [0][0] is either 1 or 0
result = classifier.predict(test_image)
training_set.class_indices
## if "result" is filled with the value 1
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)


# In[ ]:





# In[ ]:





# In[ ]:




