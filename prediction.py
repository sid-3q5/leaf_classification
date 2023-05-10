import streamlit as st
from tensorflow import keras
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf

import streamlit as st
from streamlit_option_menu import option_menu


st.set_page_config(
    page_title='Medicinal Leaf Identification', layout='centered')


def main():
    st.sidebar.title('Leaf Identify')
    menu = ['Home', 'About Us']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'Home':
        st.header('Home')
        # Add content for Home page here
    elif choice == 'About Us':
        st.header('About Us')
        st.write('Our project is focused on the automatic identification of fish species. This technology has the potential to greatly benefit humans in a number of ways.')
        st.write('For example, it can help with conservation efforts by allowing for more accurate tracking of fish populations. It can also aid in the fishing industry by allowing for more efficient and sustainable fishing practices.')
        # Add content for About Us page here

# Set background color to white
    page_bg_color = '''
    <style>
    body {
    background-color: white;
    }
    </style>
    '''
    st.markdown(page_bg_color, unsafe_allow_html=True)


if __name__ == '__main__':
    main()


# <------------------------------------------***************************************---------------------------------------------->


# Load the pre-trained model
model = keras.models.load_model('model.h5')


def preprocess_image(image):
    # Open the uploaded image using PIL
    img = Image.open(image)

    # Resize the image to the desired size
    img = img.resize((256, 256))

    # Convert the image to a NumPy array
    img_array = np.array(img)

    # Normalize the pixel values
    img_array = img_array / 255.0
    # Add a batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# Load the pre-trained model
# model = keras.models.load_model('Model.h5')

# Create a file uploader for the user to upload an image
uploaded_file = st.file_uploader(
    "Choose an image", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:

    # to print image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the uploaded image
    img = preprocess_image(uploaded_file)

    # Make a prediction using the pre-trained model
    prediction = model.predict(img)

    class_labels = ['neem', 'tulsi']

    pred = np.argmax(prediction, axis=-1)

    # Display the prediction result
    st.write(f"The leaf is: {class_labels[pred[0]]}")

    # Extract the class label and confidence score for the top predicted class
    class_index = np.argmax(prediction[0])
    # class_label = 'class_labels'  # Replace with the actual class name corresponding to class_index
    confidence_score = prediction[0][class_index]
    # st.write(f"The predicted fish species is: {class_labels[pred[0]]}")
    st.write(
        f'Top predicted class: {class_labels[pred[0]]} (confidence: {confidence_score:.2f})')
