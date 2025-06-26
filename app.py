import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import time
import pandas as pd

# App title and description
st.set_page_config(page_title="PestScanner Classification App", page_icon=":herb:")
st.title("🌱 PestScanner Disease Classification App")

# Sidebar info - Team Members
st.sidebar.header("Team Members")
st.sidebar.info(
    """
    **Abdelrahman**  
    Team Leader 
    
    **Amira**  
    Software Member
    
    **Omar**  
    Mechanical Member  
    """
)

# Green info message about what the app does - now in sidebar
st.sidebar.success("""
This app uses AI to analyze citrus leaf images and detect diseases.
Currently identifies:
- Black spot
- Citrus canker
""")

# Load your trained model (silently)
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('plant_disease_classifier.h5')
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

model = load_model()

# Define your class names
class_names = ['black-spot', 'citrus-canker']

# Image preprocessing function
def preprocess_image(image):
    img_array = np.array(image)
    if img_array.shape[-1] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    elif len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    img_array = cv2.resize(img_array, (224, 224))
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# File uploader
uploaded_file = st.file_uploader(
    "Choose a citrus leaf image...", 
    type=["jpg", "jpeg", "png"]
)

# Add guidance when no file is uploaded
if model is not None and uploaded_file is None:
    st.info("ℹ️ Please upload a citrus leaf image to get a diagnosis")

if model is not None and uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess and predict
    with st.spinner('Analyzing the leaf...'):
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        time.sleep(1)
    
    # Display results
    st.success("Analysis Complete!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Diagnosis")
        disease = class_names[predicted_class]
        st.write(f"**Detected Disease:** {disease}")
        st.write(f"**Confidence:** {confidence:.2%}")
        st.warning("⚠️ Disease detected!")
    
    with col2:
        st.subheader("Probability Distribution")
        chart_data = pd.DataFrame({
            "Disease": class_names,
            "Probability": predictions[0]
        })
        st.bar_chart(chart_data.set_index('Disease'))

# Footer
st.markdown("---")
st.markdown(
    """
    <style>
    .footer {
        font-size: 12px;
        color: #777;
        text-align: center;
    }
    </style>
    <div class="footer">
        Citrus Disease Classifier | Made with Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
