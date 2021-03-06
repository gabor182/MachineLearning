# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

# Initializing the CNN
classifier = Sequential()

# Step 1
classifier.add(Convolution2D(input_shape=(128, 128, 3), filters = 32, kernel_size=(3, 3), activation = 'relu'))
classifier.add(Convolution2D(filters = 32, kernel_size=(3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 2
classifier.add(Convolution2D(filters = 32, kernel_size=(3, 3), activation = 'relu'))
classifier.add(Convolution2D(filters = 32, kernel_size=(3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Convolution2D(filters = 32, kernel_size=(3, 3), activation = 'relu'))
classifier.add(Convolution2D(filters = 32, kernel_size=(3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(128, 128),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(128, 128),
                                            batch_size=32,
                                            class_mode='binary')

history = classifier.fit_generator(training_set,
                                   steps_per_epoch=(8000 / training_set.batch_size),
                                   epochs=22,
                                   validation_data=test_set,
                                   validation_steps=(2000 / test_set.batch_size))

history_dict = history.history

# Plot training and validation loss
import matplotlib.pyplot as plt

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)

# "bo" is for blue dot
plt.plot(epochs, loss_values, 'bo')
# "b+" is for blue cross
plt.plot(epochs, val_loss_values, 'b+')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.show()

# Plot training and validation accuracy
plt.clf() # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
plt.plot(epochs, acc_values, 'bo')
plt.plot(epochs, val_acc_values, 'b+')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.show()