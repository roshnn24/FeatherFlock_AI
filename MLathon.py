import tensorflow as tf
import os

import pathlib
import numpy as np
data_dir = pathlib.Path("/Users/rosh/Downloads/Train_data")
class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))
class_names = list(class_names)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
def view_random_image(target_dir, target_class):

  target_folder = target_dir + "/" + target_class

  random_image = random.sample(os.listdir(target_folder), 1)

  img = mpimg.imread(target_folder + "/" + random_image[0])
  plt.imshow(img)
  plt.title(target_class)
  plt.axis("off")

  print(f"Image shape: {img.shape}")
  plt.show()
  return img


#img = view_random_image(target_dir="/Users/rosh/Downloads/Train_data",target_class)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.random.set_seed(42)


# Define data augmentation parameters
train_datagen = ImageDataGenerator(
    rotation_range=20,      # Random rotation in the range [-20, 20] degrees
    width_shift_range=0.1,  # Random horizontal shift by up to 10% of the width
    height_shift_range=0.1, # Random vertical shift by up to 10% of the height
    shear_range=0.2,        # Shear intensity (shear angle in radians)
    zoom_range=0.2,         # Random zoom in the range [0.8, 1.2]
    horizontal_flip=True,   # Random horizontal flipping
    vertical_flip=True,     # Random vertical flipping
    fill_mode='nearest',    # Fill mode for points outside the input boundaries
    rescale=1./255          # Rescaling factor
)

valid_datagen = ImageDataGenerator(
    rescale=1./255          # Rescaling factor
)


train_dir = "/Users/rosh/Downloads/Train_data"
valid_dir = "/Users/rosh/Downloads/Validation_data"

train_data = train_datagen.flow_from_directory(directory=train_dir,
                                               batch_size=32,
                                               target_size=(224, 224),
                                               class_mode="categorical",
                                               seed=42)
valid_data = valid_datagen.flow_from_directory(directory=valid_dir,
                                               batch_size=32,
                                               target_size=(224, 224),
                                               class_mode="categorical",
                                               seed=42)


model_1 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu", input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPool2D(pool_size=2, padding="valid"),
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])


model_1.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])


history = model_1.fit(train_data,
                      epochs=100,
                      steps_per_epoch=len(train_data),
                      validation_data=valid_data,
                      validation_steps=len(valid_data),
                      verbose=1)

model_1.save("model_3.h5")
#
