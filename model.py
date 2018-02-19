
# coding: utf-8

# ![CNN architecture](CNN architecture.jpg)
# ![NN training flow](NN training flow.JPG)
# 

# In[8]:


import csv
import cv2
import numpy as np

import matplotlib.image as mpimg
import matplotlib
import matplotlib.pyplot as plt
#import zipfile
import random
#zip = zipfile.ZipFile('./data.zip')

#zip.extractall('data1')
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Read train data

# In[13]:


lines = []

with open('./train/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

lines_rand = random.sample(lines,2000)

del lines

lines = []

lines = lines_rand

del lines_rand

len(lines)


# ## Import train data

# In[14]:


#print(len(lines))


# ## Data Augument 1: convert original bgr channels to RGB and YUV, and keep all the images. In this way, we augument data to 3 times.

# In[15]:


images = []
measurements = []
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('\\')[-1]
    #print(filename)
        current_path = './train/IMG/' + filename
        image = cv2.imread(current_path)
    #print(image.shape)
        images.append(image)
        img_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        img_yuv = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
        images.append(img_rgb)
        images.append(img_yuv)   
    #apply correction of 0.2 to left and right images
    
    # i = 0,1,2 for center, left and right, respectively
        if i == 0:
            measurement = float(line[3])
        else:
            measurement = float(line[3]) + (-0.2)*i + 0.3
    #print(measurement)
        measurements.append(measurement)
        measurements.append(measurement)
        measurements.append(measurement)


#X_train = np.array(images)
#y_train = np.array(measurements)
print(len(images),len(measurements))


# In[16]:


fig = plt.figure(figsize=(15,10)) 
fig.add_subplot(2,2,1)
plt.title(measurements[3])
plt.imshow(images[3])

plt.subplot(2,2,2)
plt.title(measurements[4])
plt.imshow(images[4])

plt.subplot(2,2,3)
plt.title(measurements[5])
plt.imshow(images[5])


# ## Data Pre-processing (normalization, regularization, flip images etc)

# ### Note: matplotlib complaint about the negative values of pixels in the image, as a consequence I had to rescale the vaules to 0 to 1, instead of -0.5 to 0.5.

# ### Flip images

# In[17]:


#img0 = cv2.flip(X_train[0],0)
img1 = cv2.flip(images[0],1)
#img2 = cv2.flip(X_train[0],-1)

fig = plt.figure(figsize=(15,10)) 
fig.add_subplot(2,2,1)
plt.title('0')
plt.imshow(images[0])

plt.subplot(2,2,2)
plt.title('2')
plt.imshow(img1)


# In[18]:


## flip all the images taken by left and right cameras. 
X_train = []
y_train = []
for i in range(len(images)):
    X_flip = cv2.flip(images[i],1)
    y_flip = -1 * measurements[i]
    X_train.append(images[i])
    X_train.append(X_flip)
    y_train.append(measurements[i])
    y_train.append(y_flip)

len(X_train)


# In[19]:


X_train = np.array(X_train)
y_train = np.array(y_train)


# ## Set up keras

# In[20]:


from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Activation, Dropout, Cropping2D, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.utils import shuffle


#X_train, y_train = shuffle(X_train, y_train)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((60,25),(0,0)), input_shape = (160,320,3)))

model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.8))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10, activation = 'elu'))
model.add(Dense(1))
model.summary()

model.compile(loss = 'mse', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 5)

model.save('model.h5')

