# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import numpy as np
# import tensorflow as tf
# valid_datagen = ImageDataGenerator(
#     rescale=1./255          # Rescaling factor
# )
# valid_dir = "/Users/rosh/Downloads/Validation_data"
# valid_data = valid_datagen.flow_from_directory(directory=valid_dir,
#                                                batch_size=32,
#                                                target_size=(224, 224),
#                                                class_mode="categorical",
#                                                seed=42)
# loaded_model = tf.keras.models.load_model('improved_model_4.h5')
# true_labels = []
# for i in range(len(valid_data)):
#     _, labels = valid_data[i]
#     true_labels.extend(np.argmax(labels, axis=1))
#
# # Print true labels
# print("True labels:", true_labels)
# pred_prob = loaded_model.predict(valid_data)
# preds = pred_prob.argmax(axis=1)
# print("Predicted: ")
# count = 0
# for i in range(len(preds)):
#     if true_labels[i] == preds[i]:
#         count += 1
# print(count)
#print(tf.keras.models.load_model('model_4_improved_1.h5').summary())
import keras
import tensorflow as tf

print("Keras version:", keras.__version__)
print("TensorFlow version:", tf.__version__)
