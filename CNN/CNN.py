from __future__ import print_function
from aura.aura_loader import read_file
import numpy as np
from aura.extractor_util import reshape
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

print("Modules imported.")

# Set up data
cancerous_train_data = read_file(path="../../Aura_Data/ChunkedRIDER/{256x256x3511}Chunk0.aura").T
healthy_train_data = read_file(path="../../Aura_Data/ChunkedHealthy/{136x136x2353}Chunk0.aura")
healthy_train_data = reshape(healthy_train_data, (256,256,2353)).T
train_data = np.zeros((5864, 256,256))
for i in range(3511):
    train_data[i] = cancerous_train_data[i]
for i in range(2353):
    train_data[i + 3511] = healthy_train_data[i]
print(train_data.shape)

cancerous_test_data = read_file(path="../../Aura_Data/ChunkedRIDER/{256x256x3511}Chunk1.aura").T
healthy_test_data = read_file(path="../../Aura_Data/ChunkedHealthy/{136x136x2353}Chunk1.aura")
healthy_test_data = reshape(healthy_test_data, (256,256,2353)).T
test_data = np.zeros((5864, 256,256))
for i in range(3511):
    test_data[i] = cancerous_test_data[i]
for i in range(2353):
    test_data[i + 3511] = healthy_test_data[i]
print(test_data.shape)

labels = np.zeros(5864)
for i in range(3511):
    labels[i] = 1

batch_size = 733
num_classes = 2
epochs = 8

# input image dimensions
img_rows, img_cols = 256, 256

y_train = labels.copy()
y_test = labels.copy()

x_train = train_data.reshape(5864,256,256,1)
x_test = test_data.reshape(5864,256,256,1)


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

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
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])