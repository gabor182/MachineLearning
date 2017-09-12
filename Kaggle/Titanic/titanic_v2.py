# K-Fold Cross Validation
# Final accuracy on test set: 77.51%

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras import models
from keras import layers
from keras import regularizers

# Building model
def build_model(input_width):
    model = models.Sequential()
    model.add(layers.Dense(16, input_shape=(input_width, ), activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics = ['accuracy'])
    return model;

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

#K-Fold Cross validation
k = 4
num_val_samples = int(len(x_train) / k)
num_epochs = 50
all_acc_histories = []
all_loss_histories = []
for i in range(k):
    print('processing fold #', i + 1)
    # Prepare the validation data: data from partition # k
    val_data = x_train[i * num_val_samples : (i + 1) * num_val_samples, :]
    val_targets = y_train[i * num_val_samples : (i + 1) * num_val_samples]

    # Prepare the training data: data from all other partitions
    partial_train_data = np.concatenate([x_train[:i * num_val_samples, :], 
                                         x_train[(i + 1) * num_val_samples:, :]])
    partial_train_targets = np.concatenate([y_train[:i * num_val_samples], 
                                            y_train[(i + 1) * num_val_samples:]])

    # Build the Keras model (already compiled)
    model = build_model(len(partial_train_data[0]))
    # Train the model (in silent mode, verbose=0)
    history = model.fit(partial_train_data, partial_train_targets, 
                        batch_size=16,
                        epochs=num_epochs,
                        validation_split=0.20,
                        verbose=0)
    
    acc_history = history.history['val_acc']
    loss_history = history.history['val_loss']
    all_acc_histories.append(acc_history)
    all_loss_histories.append(loss_history)

average_acc_history = [np.mean([x[i] for x in all_acc_histories]) for i in range(num_epochs)]
average_loss_history = [np.mean([x[i] for x in all_loss_histories]) for i in range(num_epochs)]

# Plotting validation scores

plt.plot(range(len(average_acc_history)), average_acc_history)
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.grid()
plt.show()

plt.clf()

plt.plot(range(len(average_loss_history)), average_loss_history)
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.grid()
plt.show()

predictions = model.predict(x_test)
predictions = (predictions > 0.5).astype(int)
result_dataframe = pd.DataFrame(data=predictions, index=test_dataset.iloc[:, 0].values, columns=['Survived'])

result_dataframe.to_csv('predictions_v2.csv', index=True, header=True, index_label='PassengerId')