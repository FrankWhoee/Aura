from __future__ import print_function
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from aura.aura_loader import get_data
import matplotlib.pyplot as plt
from time import time

print("Modules imported.")
print("Current Working Directory" + os.getcwd())

# Prepare paths for GCP training
root = "../Aura_Data/"
train_paths = [root + "{136x136x16588}HealthyTrainset.aura",
               root + "{256x256x7918}BTPTrainset.aura", root + "{256x256x15624}LGGTrainset.aura", root + "{256x256x21994}CPTACTrainset.aura"]
test_paths = [root + "{136x136x5529}HealthyTestset.aura",
              root + "{256x256x879}BTPTestset.aura", root + "{256x256x1735}LGGTestset.aura", root + "{256x256x7331}CPTACTestset.aura"]

# Prepare paths for local training experimentation
# root = "../Aura_Data/Chunked/Dataset/"
# train_paths = [root + "{136x136x181}HealthyTrainset.aura", root + "{256x256x270}CancerTrainset.aura"]
# test_paths = [root + "{136x136x181}HealthyTestset.aura", root + "{256x256x270}CancerTestset.aura"]

train_data, train_label = get_data(train_paths)
test_data, test_label = get_data(test_paths)

# Merge labels to combine databases
for i,label in enumerate(train_label):
    if label == 2:
        train_label[i] = 0
    if label == 3:
        train_label[i] = 1

# Merge labels to combine databases
for i,label in enumerate(test_label):
    if label == 2:
        test_label[i] = 0
    if label == 3:
        test_label[i] = 1

train_n, train_l, train_w = train_data.shape
test_n, test_l, test_w = test_data.shape

# Set up CNN
batch_size = 32
num_classes = 2
epochs = 10
# input image dimensions
img_rows, img_cols = train_l, train_w

y_train = train_label.copy()
y_test = test_label.copy()

x_train = train_data.reshape(train_n, train_l, train_w, 1)
x_test = test_data.reshape(test_n, test_l, test_w, 1)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(str(num_classes) + " classes set.")
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

# Convolutional layers and Max pooling
model.add(Conv2D(32, kernel_size=(16, 16),
                 activation='relu',
                 input_shape=(train_l, train_w, 1)))
model.add(MaxPooling2D(pool_size=(8, 8)))
model.add(Conv2D(64, (4, 4), activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Conv2D(128, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (2, 2), activation='relu'))

# Dense layers and output
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.05))
model.add(Dense(2048, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.21))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# checkpoint
filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history = model.fit(x_train, y_train,
                    callbacks=callbacks_list,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
finish_time = str(time.time())
model.save("model" + finish_time[:finish_time.find(".")] + ".hf")
print('Test loss:', score[0])
print('Test accuracy:', score[1])

plt.plot(history.history['loss'])
print(history.history['loss'])
print(history.history['val_loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
