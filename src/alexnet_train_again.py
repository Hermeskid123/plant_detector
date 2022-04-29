'''
https://tensorflow.google.cn/tutorials/load_data/images?hl=en
''' and  None


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from tensorflow.keras.layers import *


print("tf version ",tf.__version__)
  



def main():
    batch_size = 50
    img_height = 180
    img_width = 180


    train_data = keras.utils.image_dataset_from_directory("plantnet_300K/images_train", image_size=(img_height, img_width),batch_size=batch_size,color_mode="rgb")
    val_data = keras.utils.image_dataset_from_directory("plantnet_300K/images_val", image_size=(img_height, img_width),batch_size=batch_size,color_mode="rgb")
    test_data = keras.utils.image_dataset_from_directory("plantnet_300K/images_test", image_size=(180, 180),batch_size=batch_size,color_mode="rgb")
    class_names = train_data.class_names

    print("number of classes ",len(class_names))
     
    
    for image_batch, labels_batch in train_data:
        print("image batch shape ",image_batch.shape)
        print("lables_batch shape ",labels_batch.shape)
        break

    normalization_layer = tf.keras.layers.Rescaling(1./255)
    normalized_ds = train_data.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    print(np.min(first_image), np.max(first_image))
    

    
    num_classes = len(class_names)
    AlexNet = keras.models.load_model("AlexNetModelv3_250_batch_2_epochs.keras")
    AlexNet.summary()
    filename = "AlexNetModelv4_50_batch_15_epochs_rgb_nearest.keras"

    AlexNet.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    callbacks = [keras.callbacks.ModelCheckpoint(filename, save_best_only=True)]
    AlexNet.fit(
        train_data,
        validation_data=val_data,
        epochs=15,
        callbacks = callbacks
        )
    
    AlexNet.save(filename)

    (test_loss, test_acc) = AlexNet.evaluate(test_data)

    print("Test accuracy: %.2f%%" % (test_acc*100))

    print("End of main")

if __name__ == "__main__":
    main()

