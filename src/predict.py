import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from tensorflow.keras.layers import *
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
import os 
import sys 

def load_image(img_path):
    img = image.load_img(img_path, target_size=(180, 180))
    img_tensor = image.img_to_array(img)                                               
    img_tensor = np.expand_dims(img_tensor, axis=0)                                                                                                                                    
    img_tensor /= 255.                                                                                 
    return img_tensor

filename = sys.argv[1]
pic = sys.argv[2]
model = keras.models.load_model(filename)
image = load_image(pic)
model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
    )

result = model.predict(image)[0]
print (result.shape)
proba = np.max(result)
label = str(np.where(result==proba)[0])
label = "{}: {:.2f}%".format(label, proba * 100)
print(label)
