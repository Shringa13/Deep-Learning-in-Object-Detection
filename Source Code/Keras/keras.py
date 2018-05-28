# Importing Pandas and Numpy
import pandas as pd 
import numpy as np

#Importing Keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator 
from keras.callbacks import TensorBoard

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=False)
# Initialising classifiers
cnn_classifier = Sequential()

# Convolution method
"""
32 - Number of feature detectors 
3  - Rows of Feature Detectors
3  - Columns of Feature Detectos 
input shape -  To ensure all the input images are of the same format. Here 64,64 represents the image matrix and 3 represents it 
               is a colour image. We have the last parameter of input_shape to 1 if we are dealing with a black and white image.
Activation function - Relu 
"""
cnn_classifier.add(Convolution2D(32,(3,3),input_shape=(64,64,3),activation= 'relu'))

# Pooling method 
"""
pool_size - This is used to reduce the size of the feature maps by 2. 
"""
cnn_classifier.add(MaxPooling2D(pool_size = [2,2]))

# Flattening method 
"""
Flatten() - To convert the pooling output to one dimensional matrix. This matrix will contain pixel pattern of then input image. 
            
"""
cnn_classifier.add(Flatten())

# Full Connection method
"""
The single matrix from Full Connection is fed into an conventional neural network ( A fully connected NN).
Dense - Creating 128 nodes for the hidden layers and creating a single output node. 
"""
cnn_classifier.add(Dense(activation= 'relu',units=128))
cnn_classifier.add(Dense(activation = 'sigmoid',units= 1))

# Compiling CNN 

cnn_classifier.compile(optimizer='adam', loss= 'binary_crossentropy', metrics=['accuracy'])

# Image preprocessing. Source : Keras API documentation  
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory('//home//shringa//Downloads//CNN//Convolutional_Neural_Networks//Convolutional_Neural_Networks//training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

test_data = test_datagen.flow_from_directory('//home//shringa//Downloads//CNN//Convolutional_Neural_Networks//Convolutional_Neural_Networks//test_set',
                                             target_size=(64, 64),
                                             batch_size=32,
                                             class_mode='binary')

# Fitting data and Predicting images from test set.
"""
Number of epochs = 25, So training of 8000 images will happen over 25 epochs.
one epoch = one forward pass and one backward pass of all the training examples
batch size = The number of training examples in one forward/backward pass. 
"""
cnn_classifier.fit_generator(train_data,
                    steps_per_epoch=25000,
                    epochs=25,
                    validation_data=test_data,
                    validation_steps=10000,
                    callbacks=[tensorboard])