import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load pre-trained model
model = tf.keras.models.load_model('mnist_cnn.h5')

# Streamlit app
st.title('MNIST Digit Classifier')
st.write('Upload a handwritten digit image (28x28, grayscale).')

# File uploader
uploaded_file = st.file_uploader('Choose an image...', type=['png', 'jpg'])
if uploaded_file is not None:
    # Preprocess image
    image = Image.open(uploaded_file).convert('L')
    image = image.resize((28, 28))
    image_array = np.array(image).reshape(1, 28, 28, 1).astype('float32') / 255.0

    # Predict
    prediction = model.predict(image_array)
    predicted_digit = np.argmax(prediction)

    # Display results
    st.image(image, caption='Uploaded Image', width=200)
    st.write(f'Predicted Digit: {predicted_digit}')