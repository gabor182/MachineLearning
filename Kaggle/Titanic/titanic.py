# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras import models
from keras import layers

# Data preprocessing
train_dataset = pd.read_csv('dataset/train.csv')
x_train = train_dataset.iloc[:, [2, 4, 5, 6, 7]].values
y_train = train_dataset.iloc[:, 1].values;

# Taking care of missing data
from sklearn.preprocessing import Imputer, LabelEncoder, StandardScaler

imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
x_train[:, 2:5] = imputer.fit_transform(x_train[:, 2:5])

labelencoder = LabelEncoder()
x_train[:, 1] = labelencoder.fit_transform(x_train[:, 1])

sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)

# Building the network

model = models.Sequential()
model.add(layers.Dense(32, input_shape=(8,), activation='relu'))