# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras import models
from keras import layers

# Data preprocessing

train_dataset = pd.read_csv('dataset/train.csv')
test_dataset = pd.read_csv('dataset/test.csv')
x_train = train_dataset.iloc[:, [2, 4, 5, 6, 7, 9]].values
y_train = train_dataset.iloc[:, 1].values;
x_test = test_dataset.iloc[:, [1, 3, 4, 5, 6, 8]].values

from sklearn.preprocessing import Imputer, LabelEncoder, StandardScaler

# Taking care of missing data
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
x_train[:, 2:6] = imputer.fit_transform(x_train[:, 2:6])
x_test[:, 2:6] = imputer.fit_transform(x_test[:, 2:6])

# Encoding sex
labelencoder = LabelEncoder()
x_train[:, 1] = labelencoder.fit_transform(x_train[:, 1])
x_test[:, 1] = labelencoder.transform(x_test[:, 1])

x_train = x_train.astype(float, copy=False)
x_test = x_test.astype(float, copy=False)

# Scale data to around zero
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)


# Building the network
model = models.Sequential()
model.add(layers.Dense(32, input_shape=(6,), activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics = ['accuracy'])
history = model.fit(x_train, y_train, 
                    batch_size=32,
                    epochs=38,
                    validation_split=0.2)

# Calculate the loss and accuracy for a given data set
#results = model.evaluate(x_test, y_test)

history_dict = history.history

# Plot training and validation loss

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

predictions = model.predict(x_test)
predictions = (predictions > 0.5).astype(int)
result_dataframe = pd.DataFrame(data=predictions, index=test_dataset.iloc[:, 0].values, columns=['Survived'])

result_dataframe.to_csv('predictions.csv', index=True, header=True, index_label='PassengerId')