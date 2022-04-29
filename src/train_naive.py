'''
https://tensorflow.google.cn/tutorials/load_data/images?hl=en
''' and  None


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
print("tf version ",tf.__version__)


def main():
    batch_size = 100
    img_height = 180
    img_width = 180


    train_data = keras.utils.image_dataset_from_directory("plantnet_300K/images_train", image_size=(img_height, img_width),batch_size=batch_size)
    val_data = keras.utils.image_dataset_from_directory("plantnet_300K/images_val", image_size=(img_height, img_width),batch_size=batch_size)
    test_data = keras.utils.image_dataset_from_directory("plantnet_300K/images_test", image_size=(img_height, img_width),batch_size=batch_size)
    
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

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
        ])

    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=10
        )
    
    model.save("Preston_ModelV3.keras")

    (test_loss, test_acc) = model.evaluate(test_data)

    print("Test accuracy: %.2f%%" % (test_acc*100))

    print("End of main")

if __name__ == "__main__":
    main()

