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
model = keras.models.load_model('leaf_classification_model.h5')


def predict_leaf(image):
    img = Image.open(image)
    img = img.resize((256, 256))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict the class probabilities
    prediction = model.predict(img_array)
    class_labels = ['neem', 'tulsi']

    # Get the predicted class index
    pred_index = np.argmax(prediction, axis=1)[0]
    pred_label = class_labels[pred_index]
    confidence_score = prediction[0][pred_index]

    return pred_label, confidence_score, pred_index, class_labels, prediction


uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Call the predict_leaf function
    pred_label, confidence_score, pred_index, class_labels, prediction = predict_leaf(uploaded_file)
    st.write(f"prediction: {prediction}")
    st.write(f"class_labels: {class_labels}")
    st.write(f"pred_index: {pred_index}")
    st.write(f"The leaf is: {pred_label}")
    st.write(f"Confidence score: {confidence_score:.2f}")

