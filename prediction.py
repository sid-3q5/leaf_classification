import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow import keras
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu


# Load the trained model
model = load_model("leaf_classification_model.h5")

# Set the class labels
class_labels = ['neem', 'tulsi']

# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to the required input shape of the model
    image = image.resize((256, 256))
    # Convert image to numpy array
    img_array = np.array(image)
    # Normalize pixel values
    img_array = img_array / 255.0
    # Expand dimensions to match the input shape of the model
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions
def make_prediction(image):
    # Preprocess the image
    img_array = preprocess_image(image)
    # Make the prediction
    predictions = model.predict(img_array)
    # Get the predicted class index
    predicted_class_index = np.argmax(predictions[0])
    # Get the predicted class label
    predicted_class = class_labels[predicted_class_index]
    # Get the confidence score
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence, predictions, predicted_class_index

# Streamlit app
def main():
    # Set app title
    st.title("Leaf Classification")

    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image file
        image = Image.open(uploaded_file)
        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Make predictions when the user clicks the "Predict" button
        if st.button("Predict"):
            # Make the prediction
            predicted_class, confidence = make_prediction(image)
            # Display the predicted class and confidence score
            st.write("Predicted Class:", predicted_class)
            st.write("Confidence:", confidence, "%")
            st.write("predictions:", predictions)
            st.write("predicted_class_index:", predicted_class_index)

# Run the app
if __name__ == '__main__':
    main()








