import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from tensorflow.keras.layers import *



import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
import os 
import sys 

filename = sys.argv[1]

test_data = keras.utils.image_dataset_from_directory("plantnet_300K/images_test", image_size=(180, 180),batch_size=180)
labels = np.concatenate([y for x, y in test_data], axis=0)
model = keras.models.load_model(filename)
# The next line calls your test_model function.
model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
    )

model.summary()
(test_loss, test_acc) = model.evaluate(test_data)

print("test accuracy: %.1f%%" % (test_acc*100))
