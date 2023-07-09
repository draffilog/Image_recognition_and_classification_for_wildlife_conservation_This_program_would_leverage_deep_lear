import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

# Load pre-trained ResNet50 model
model = tf.keras.applications.ResNet50(weights='imagenet')

# Function to classify image
def classify_image(image_path):
    # Load and preprocess image
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Classify image
    preds = model.predict(x)
    decoded_preds = decode_predictions(preds, top=3)[0]

    # Print top predictions
    for _, label, probability in decoded_preds:
        print(f"{label}: {probability * 100:.2f}%")

# Example usage
image_path = 'path_to_image.jpg'
classify_image(image_path)