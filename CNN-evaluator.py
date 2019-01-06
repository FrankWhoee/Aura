from __future__ import print_function
import time
import os
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy
from matplotlib import pyplot as plt

from aura import aura_loader as al
from aura import extractor_util as eu

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(256,256,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.load_weights("Model-v1.hf")

images = al.read_file("../Aura_Data/ChunkedHealthyTestset/{136x136x181}Chunk0.aura").T
image = np.zeros((1,256,256,1))
image[0] = eu.convertToSize(images[79],(256,256)).reshape(256,256,1)

plt.imshow(images[79])
plt.show()

print(model.predict(image))