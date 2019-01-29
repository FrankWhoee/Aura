from __future__ import print_function
import time
import os
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import random
import scipy.misc
import matplotlib.pyplot as plt
from aura.extractor_util import reshape
from aura.extractor_util import parseAuraDimensions as pAD
from aura.aura_loader import read_file

print("Modules imported.")
print(os.getcwd())
root = "../Aura_Data/";
cancerPath = root + "Dataset/"
healthyPath = root + "Dataset/"
tumorPath = root + "Dataset/"

# cancerSize = "{256x256x3511}"
# healthySize = "{136x136x2353}"
# tumorSize = "{256x256x5501}"

cancerSize = "{256x256x270}"
healthySize = "{136x136x181}"
tumorSize = "{256x256x250}"

trainSuffix = "Trainset"
testSuffix = "Testset"
cancerPrefix = "Cancer"
healthyPrefix = "Healthy"
tumorPrefix = "Tumor"
fileExtension = ".aura"

trainTumor = False

cl,cw,cn = pAD(cancerSize)
hl,hw,hn = pAD(healthySize)
tl,tw,tn = pAD(tumorSize)
fl, fw = max(cl, cw, hl, hw), max(cl, cw, hl, hw)
fn = cn + hn
if trainTumor:
    fn += tn
# Set up data

# Load training data
cancerous_train_data = read_file(path=cancerPath + cancerSize + cancerPrefix + trainSuffix + fileExtension).T
healthy_train_data = read_file(path=healthyPath+ healthySize + healthyPrefix + trainSuffix + fileExtension)
healthy_train_data = reshape(healthy_train_data, (fl,fw,hn)).T
if trainTumor:
    tumor_train_data = read_file(path=tumorPath + tumorSize + tumorPrefix + trainSuffix + fileExtension).T

# Load testing data
cancerous_test_data = read_file(path=cancerPath + cancerSize + cancerPrefix + testSuffix + fileExtension).T
healthy_test_data = read_file(path=healthyPath + healthySize + healthyPrefix + testSuffix + fileExtension)
healthy_test_data = reshape(healthy_test_data, (fl,fw, hn)).T
if trainTumor:
    tumor_test_data = read_file(path=tumorPath + tumorSize + tumorPrefix + testSuffix + fileExtension).T

# Compile training data into one array
train_data = np.zeros((fn, fl,fw))
for i in range(cn):
    train_data[i] = cancerous_train_data[i]
for i in range(hn):
    train_data[i + cn] = healthy_train_data[i]
if trainTumor:
    for i in range(tn):
        train_data[i + cn + hn] = tumor_train_data[i]
print(train_data.shape)

# Compile testing data into one array
test_data = np.zeros((fn, fl,fw))
for i in range(cn):
    test_data[i] = cancerous_test_data[i]
for i in range(hn):
    test_data[i + cn] = healthy_test_data[i]
if trainTumor:
    for i in range(tn):
        test_data[i + cn + hn] = tumor_test_data[i]
print(test_data.shape)

# Label training data
training = []
for i in range(cn):
    training.append([train_data[i], 1])
for i in range(hn):
    training.append([train_data[i + cn], 0])
if trainTumor:
    for i in range(tn):
        training.append([train_data[i + cn + hn], 2])

# Label testing data
testing = []
for i in range(cn):
    testing.append([test_data[i], 1])
for i in range(hn):
    testing.append([test_data[i + cn], 0])
if trainTumor:
    for i in range(tn):
        testing.append([test_data[i + cn + hn], 2])

# Shuffle training data
random.shuffle(training)

# Shuffle testing data
random.shuffle(testing)

# Separate training images and labels
train_label = np.zeros(fn)
train_data = np.zeros(train_data.shape)
for i,(data,label) in enumerate(training):
    train_data[i] = data
    # plt.imshow(data)
    # plt.show()
    train_label[i] = label

# Separate testing images and labels
test_label = np.zeros(fn)
test_data = np.zeros(test_data.shape)
for i,(data,label) in enumerate(testing):
    test_data[i] = data
    test_label[i] = label

# Set up CNN
batch_size = 2
if trainTumor == True:
    num_classes = 3
else:
    num_classes = 2
epochs = 1
# input image dimensions
img_rows, img_cols = fl,fw

y_train = train_label.copy()
y_test = test_label.copy()

x_train = train_data.reshape(fn,fl,fw,1)
x_test = test_data.reshape(fn,fl,fw,1)


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(str(num_classes) + " classes set.")
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

# Convolutional layers and Max pooling
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(fl,fw,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))

# Dense layers and output
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(2048, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
finish_time = str(time.time())
model.save("model"+finish_time[:finish_time.find(".")]+".hf")
print('Test loss:', score[0])
print('Test accuracy:', score[1])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()