import os
import csv
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
# Define the path to the directory containing test images
predict_dir = "/Users/rosh/Downloads/Eval_data"  # Change this to the actual path
model = tf.keras.models.load_model("model_4_improved_8.h5")
# Define class labels
class_labels = ['Crane', 'Crow', 'Egret', 'Kingfisher','Myna','Peacock','Pitta','Rosefinch','Tailorbird','Wagtail']
# Open a CSV file to write the results
with open('pred.csv', mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Name', 'Target_name','Target_num'])
    qq=0
    # Loop through each image file and make predictions
    for img_file in os.listdir(predict_dir):
        print(qq)
        # Load and preprocess the image
        img_path = '/Users/rosh/Downloads/Eval_data'+'/'+ img_file
        img = image.load_img(img_path, target_size=(224, 224))  # Ensure the target_size matches the input size of your model

        # Preprocess the image
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array,verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0]
        # Assuming train_images.class_indices is a dictionary mapping class names to indices
        class_indices = [0,1,2,3,4,5,6,7,8,9]
        class_names =['Crane', 'Crow', 'Egret', 'Kingfisher', 'Myna', 'Peacock', 'Pitta', 'Rosefinch', 'Tailorbird', 'Wagtail']
        predicted_class_name = class_names[predicted_class]

        writer.writerow([img_file[:img_file.index('.jpg')], predicted_class_name,predicted_class])
        qq+=1

print("Predictions saved to predictions.csv")