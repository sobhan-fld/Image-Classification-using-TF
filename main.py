from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras import backend
import tensorflow
import numpy as np
import cv2
import os
from keras.models import model_from_json, save_model, load_model
import shutil

print('Start')
img_width = 200
img_height = 200
batch_size = 64

# Making Dataset Ready
train_data = ImageDataGenerator(
    rescale=1. / 255, horizontal_flip=True)
train = train_data.flow_from_directory(
    'data/train',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

test_data = ImageDataGenerator(rescale=1. / 255)
valid = test_data.flow_from_directory(
    'data/validation',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

print('train steps:', len(train))
print('valid steps:', len(valid))

# TensorFlow expects the 'channels' dimension as the last dimension for conv2d
if backend.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# Model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(
    Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D((2, 2)))


model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid')) # output: 0 to 1

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

tensorflow.device('/gpu:0')

epochs = 20
print('Epochs :', epochs)
print('Training Begins:')

# model.fit(
#     train,
#     steps_per_epoch=len(train),
#     epochs=epochs,
#     validation_data=valid,
#     validation_steps=len(valid),
#     verbose=1)
#
#
# # evaluate model
# _, acc = model.evaluate(valid, steps=len(valid))
# print('> %.3f' % (acc * 100.0))


# # serialize model to JSON
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")

# Predict From loaded model :

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#
_, acc = loaded_model.evaluate(valid, steps=len(valid))

accuracy = acc * 100.0
print('> %.2f' % accuracy, '%')


directory = os.path.abspath('test2')
paths = os.listdir(directory)

for filename in paths:
    img = cv2.imread(directory + os.sep + filename)
    img = cv2.resize(img, (200, 200))
    img = np.reshape(img, [1, 200, 200, 3])

    preds = (loaded_model.predict(img) > 0.5).astype('int32')
    if preds == 0:
        print(filename, '= cat')
        shutil.move('test2/'+ filename, 'newcat/'+ filename)
    else:
        print(filename, '= dog')
        shutil.move('test2/'+ filename, 'newdog/'+ filename)