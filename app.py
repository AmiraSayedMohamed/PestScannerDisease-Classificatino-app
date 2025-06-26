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
        # Load TFLite model and allocate tensors
        interpreter = tf.lite.Interpreter(model_path='plant_disease_classifier_quant.tflite')
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

model = load_model()

# Define your class names
class_names = ['black-spot', 'citrus-canker']

# Recommendation database based on your uploaded images
recommendation_db = {
    'black-spot': {
        'chemical': [
            {'name': 'Ortus super', 'active': 'Fenpyroximate 5% EC', 'type': 'Contact', 'safety': 'Wear gloves and mask. Avoid application in windy conditions.'},
            {'name': 'TAK', 'active': 'Chlorpyrifos 48% EC', 'type': 'Systemic', 'safety': 'Highly toxic to bees. Apply in evening when bees are less active.'}
        ],
        'organic': [
            'Neem oil spray (apply every 7-10 days)',
            'Baking soda solution (1 tbsp baking soda + 1 tsp vegetable oil + 1 gallon water)',
            'Copper-based fungicides'
        ],
        'cultural': [
            'Remove and destroy infected leaves',
            'Improve air circulation by pruning',
            'Avoid overhead watering',
            'Rotate with non-citrus crops for 2 seasons'
        ],
        'description': 'Black spot is a fungal disease that causes dark spots on leaves and fruit. It thrives in warm, wet conditions.'
    },
    'citrus-canker': {
        'chemical': [
            {'name': 'Biomectin', 'active': 'Abamectin 3% EC', 'type': 'Systemic', 'safety': 'Use protective clothing. Do not apply near water sources.'},
            {'name': 'AVENUE', 'active': 'Imidacloprid 70% SC', 'type': 'Systemic', 'safety': 'Toxic to aquatic organisms. Keep away from waterways.'}
        ],
        'organic': [
            'Copper-based bactericides',
            'Streptomycin sulfate (antibiotic spray)',
            'Garlic and chili pepper extract sprays'
        ],
        'cultural': [
            'Remove and burn infected plants',
            'Disinfect tools with 10% bleach solution',
            'Plant resistant varieties when available',
            'Implement strict quarantine measures for new plants'
        ],
        'description': 'Citrus canker is a bacterial disease causing raised lesions on leaves, stems, and fruit. Highly contagious.'
    }
}

def predict_with_interpreter(interpreter, image):
    """Make prediction using TFLite interpreter"""
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Preprocess image
    input_data = preprocess_image(image).astype(np.float32)
    
    # Set the tensor to point to the input data
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get the output
    predictions = interpreter.get_tensor(output_details[0]['index'])
    return predictions

def display_recommendations(disease):
    """Display recommendations based on the detected disease"""
    if disease not in recommendation_db:
        st.warning("No recommendations available for this disease.")
        return
    
    data = recommendation_db[disease]
    
    st.subheader(f"🌱 {disease.replace('-', ' ').title()} Information")
    st.info(data['description'])
    
    st.markdown("---")
    
    # Chemical recommendations
    st.subheader("🧪 Chemical Control Options")
    if data['chemical']:
        chem_df = pd.DataFrame(data['chemical'])
        st.table(chem_df)
        st.warning("⚠️ Always follow pesticide label instructions and local regulations")
    else:
        st.info("No chemical recommendations available")
    
    # Organic recommendations
    st.subheader("🍃 Organic/Natural Remedies")
    for remedy in data['organic']:
        st.markdown(f"- {remedy}")
    
    # Cultural practices
    st.subheader("🌿 Cultural Practices")
    for practice in data['cultural']:
        st.markdown(f"- {practice}")
    
    # Regulatory info
    st.markdown("---")
    st.subheader("📜 Regulatory Information")
    st.info("""
    - Always check with your local agricultural extension office
    - Some pesticides may be restricted in your area
    - Follow recommended pre-harvest intervals
    """)

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
        predictions = predict_with_interpreter(model, image)
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
        
        # Add expander for recommendations
        with st.expander("🛠️ View Treatment Recommendations", expanded=True):
            display_recommendations(disease)
    
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
