# -*- coding: utf-8 -*-
"""
ML-project

"""

import numpy as np
from keras.datasets.mnist import load_data
from numpy.random import randint
import matplotlib.pyplot as plt
import tensorflow as tf

# example of loading the mnist dataset
# load the images into memory
(trainX, trainy), (testX, testy) = load_data()
# summarize the shape of the dataset
print('Train', trainX.shape, trainy.shape)
print('Test', testX.shape, testy.shape)

train_mask = np.isin(trainy, [0  , 1, 2, 8])
test_mask = np.isin(testy, [0  , 1, 2, 8])

X_train, Y_train = trainX[train_mask], np.array(trainy[train_mask] == 2)
X_test, Y_test = testX[test_mask], np.array(testy[test_mask] == 2)

# plot raw pixel data
i = randint(1, 1000)
print(i)
plt.imshow(X_test[i], cmap='gray')

# example of loading the mnist dataset
from keras.datasets.mnist import load_data
from matplotlib import pyplot
# load the images into memory
# plot images from the training dataset

for i in range(25):
	# define subplot
	pyplot.subplot(5, 5, 1 + i)
	# turn off axis
	pyplot.axis('off')
	# plot raw pixel data
	pyplot.imshow(X_train[i], cmap='gray_r')
pyplot.show()

print(X_train[7])


