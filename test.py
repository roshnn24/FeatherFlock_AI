# Create a function to import an image and resize it to be able to be used with our model
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pathlib
import numpy as np
data_dir = pathlib.Path("/Users/rosh/Downloads/Train_data")
class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))
class_names = list(class_names)
class_names.pop(0)
loaded_model = tf.keras.models.load_model('model_4_improved_8.h5')
def load_and_prep_image(filename, img_shape=224):
  """
  Reads an image from filename, turns it into a tensor
  and reshapes it to (img_shape, img_shape, colour_channel).
  """
  # Read in target file (an image)
  img = tf.io.read_file(filename)

  # Decode the read file into a tensor & ensure 3 colour channels
  # (our model is trained on images with 3 colour channels and sometimes images have 4 colour channels)
  img = tf.image.decode_image(img, channels=3)

  # Resize the image (to the same size our model was trained on)
  img = tf.image.resize(img, size = [img_shape, img_shape])

  # Rescale the image (get all values between 0 and 1)
  img = img/255.
  return img

# Adjust function to work with multi-class
def pred_and_plot(model, filename, class_names):
  """
  Imports an image located at filename, makes a prediction on it with
  a trained model and plots the image with the predicted class as the title.
  """
  # Import the target image and preprocess it
  img = load_and_prep_image(filename)

  # Make a prediction
  pred = model.predict(tf.expand_dims(img, axis=0))

  # Get the predicted class

  pred_class = class_names[pred.argmax()] # if more than one output, take the max

  # Plot the image and predicted class
  plt.imshow(img)
  plt.title(f"Prediction: {pred_class}")
  plt.axis(False)
  plt.show()

pred_and_plot(loaded_model, "/Users/rosh/Downloads/egret.jpg", class_names)
# # loaded_model.compile(loss='categorical_crossentropy',
# #                      optimizer='adam',
# #                      metrics=['accuracy'])
# # Get true labels
# valid_datagen = ImageDataGenerator(
#     rescale=1./255          # Rescaling factor
# )
# valid_dir = "/Users/rosh/Downloads/Validation_data"
# valid_data = valid_datagen.flow_from_directory(directory=valid_dir,
#                                                batch_size=32,
#                                                target_size=(224, 224),
#                                                class_mode="categorical",
#                                                seed=42)
# pred = loaded_model.predict(valid_data)
# preds = pred.argmax(axis=1)
# print(preds)