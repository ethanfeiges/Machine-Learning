## Code creates an AI model using the keras module to predict whether an image is a dog or a cat. 
## Testing and training data are images of cats and dogs (not attached to file)

# In[3]:


# Using a sequential model for neural network
from keras.models import Sequential
# Tools for flattening an image. (Creates proper setup)
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# In[6]:


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


# In[7]:


## Step 3 - Flattening: Turning 2D array of pixels (single dimension) into a single array (64 * 64 2D element -> 64^2 single array elements)
classifier.add(Flatten())

## Step 4 - Full connection. 
## Reducing the array to 128 size
classifier.add(Dense(units = 128, activation = 'relu'))

## Single output: (True/false, dog/cat)
## Sigmoid activiation either yields values of 1 or 0 (yes or no)
classifier.add(Dense(units = 1, activation = 'sigmoid'))


# In[16]:


## Compiling the classifier neural network:
## loss and metrics: How error is calculated
## optimizer: reverse propagation (adjusting weights during training)
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[9]:


## Part 2 - Fitting the Classifier neural network to the images (supervised training)
## 10,000 images total. 8,000 will be used for training, 2,000 will be used for testing
from keras.preprocessing.image import ImageDataGenerator
## If the photo is different shapes or sizes
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = .2, zoom_range = .2, horizontal_flip = True)

## batch_size: 32 images will be "batched" through the training each time.
## target_size: 
training_set = train_datagen.flow_from_directory(r"C:\Users\ejfei\Downloads\training_set\training_set", target_size = (64, 64), batch_size = 32, class_mode = 'binary')


# In[14]:


## formatting the test_data images
test_datagen = ImageDataGenerator(rescale = 1./255)

## Loading test set 
test_set = test_datagen.flow_from_directory(r"C:\Users\ejfei\Downloads\test_set\test_set", target_size = (64, 64), batch_size = 32, class_mode = 'binary')


# In[19]:


## Back propogation to allow classifier to update weights by propagating how much of the loss each node is responsible for
## loss: Penalty for bad prediction. (Perfect loss is zero)
## Epoch: We will go through the whole data set 10 times
## steps_per_epoch: We will look at each picture during each epoch 4000 times
## validation_data: test_set will validate the training set and evaluate the loss
## validation-steps: 10 images from the test_set for each validation
## validation: data that has never been seen before by the classifier
## accuracy: The accuray of what the classifier is training
## val_accc: The accuracy of the testing data
classifier.fit(training_set, steps_per_epoch = 4000, 
                         epochs = 10, 
                         validation_data = test_set, 
                         validation_steps = 10)
