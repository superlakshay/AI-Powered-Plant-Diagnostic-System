import numpy as np
import tensorflow as tf
import streamlit as st
import os
import json
import requests
import cv2
from PIL import Image
from datetime import datetime

# Paths
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# Load Model
model = tf.keras.models.load_model(model_path)

# Function to preprocess image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype("float32") / 255.0
    return img_array, img

# Prediction function
def predict_image_class(model, image_path, class_indices):
    preprocessed_image, original_img = load_and_preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    confidence = np.max(prediction) * 100
    return predicted_class_name, confidence, preprocessed_image, original_img

# Improved visualization - highlights diseased areas specifically
def highlight_diseased_areas(image_path):
    # Load the image
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    
    # Convert to different color spaces for better segmentation
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    
    # Define color ranges for common disease symptoms
    # Yellow/brown areas (common in many plant diseases)
    lower_yellow = np.array([20, 50, 50])
    upper_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Brown/dark areas
    lower_brown = np.array([10, 50, 20])
    upper_brown = np.array([20, 200, 200])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    
    # Black/dead tissue areas (in LAB space)
    l_channel = lab[:,:,0]
    black_mask = cv2.inRange(l_channel, 0, 50)
    
    # Combine all masks
    combined_mask = cv2.bitwise_or(yellow_mask, brown_mask)
    combined_mask = cv2.bitwise_or(combined_mask, black_mask)
    
    # Clean up the mask
    kernel = np.ones((5,5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours of diseased areas
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create visualization - highlight diseased areas in red
    visualization = img_array.copy()
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Only show larger areas
            cv2.drawContours(visualization, [contour], -1, (255, 0, 0), 2)
            # Fill the area with transparent red
            mask = np.zeros_like(visualization)
            cv2.drawContours(mask, [contour], -1, (255, 0, 0), -1)
            visualization = cv2.addWeighted(visualization, 1, mask, 0.3, 0)
    
    # Calculate disease percentage
    total_pixels = img_array.shape[0] * img_array.shape[1]
    disease_pixels = np.sum(combined_mask > 0)
    disease_percentage = (disease_pixels / total_pixels) * 100
    
    return Image.fromarray(visualization), disease_percentage

# Disease severity assessment
def assess_disease_severity(image_path):
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # Define ranges for unhealthy tissue (yellow/brown)
    lower_brown = np.array([10, 50, 50])
    upper_brown = np.array([30, 255, 255])
    
    # Create mask and calculate percentage
    mask = cv2.inRange(hsv, lower_brown, upper_brown)
    severity_percentage = np.sum(mask > 0) / (img_array.shape[0] * img_array.shape[1]) * 100
    
    if severity_percentage < 5:
        return "Mild", severity_percentage
    elif severity_percentage < 20:
        return "Moderate", severity_percentage
    else:
        return "Severe", severity_percentage

# Chatbot function using OpenRouter API
def ask_plant_disease_chatbot(question, disease_context=""):
    # OpenRouter API endpoint
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    # Your OpenRouter API key (you can get a free one at https://openrouter.ai/)
    api_key = st.secrets.get("OPENROUTER_API_KEY", "")  # Store your key in Streamlit secrets
    
    # Prepare the prompt with disease context if available
    if disease_context:
        system_message = f"You are a plant disease expert. The user has a plant with: {disease_context}. Provide helpful advice about prevention, treatment, and care."
    else:
        system_message = "You are a plant disease expert. Provide helpful advice about plant diseases, prevention, treatment, and care."
    
    # Prepare the payload
    payload = {
        "model": "google/gemini-flash-1.5",  # You can change this to any free model on OpenRouter
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": question}
        ]
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Sorry, I couldn't process your request at the moment. Error: {str(e)}"

# ---- Streamlit UI ----
st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌱", layout="wide")

# Custom CSS for clean design
st.markdown("""
    <style>
    /* Main background */
    .main {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 25%, #a5d6a7 50%, #81c784 75%, #66bb6a 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #333;
    }
    
    /* Title styling */
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #2e7d32;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    /* Section headers */
    .section-header {
        background: #4caf50;
        color: white;
        padding: 0.7rem 1rem;
        border-radius: 8px;
        margin: 1rem 0 0.8rem 0;
        font-size: 1rem;
        font-weight: 600;
    }
    
    /* Card styling */
    .card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #4caf50;
    }
    
    /* Buttons */
    .stButton > button {
        background: #4caf50;
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 6px;
        font-size: 0.9rem;
        font-weight: 600;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: #388e3c;
    }
    
    /* Success message */
    .stSuccess {
        background: #e8f5e9;
        border-left: 4px solid #4caf50;
        color: #2e7d32;
        padding: 0.7rem;
        border-radius: 6px;
        margin: 0.7rem 0;
        font-size: 0.9rem;
    }
    
    /* Info message */
    .stInfo {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        color: #1565c0;
        padding: 0.7rem;
        border-radius: 6px;
        margin: 0.7rem 0;
        font-size: 0.9rem;
    }
    
    /* Warning message */
    .stWarning {
        background: #fff4e5;
        border-left: 4px solid #f59e0b;
        color: #92400e;
        padding: 0.7rem;
        border-radius: 6px;
        margin: 0.7rem 0;
        font-size: 0.9rem;
    }
    
    /* Chat container */
    .chat-container {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        height: 400px;
        overflow-y: auto;
        margin-bottom: 0.5rem;
        border: 1px solid #e0e0e0;
    }
    
    /* Chat messages */
    .chat-message {
        padding: 0.7rem 0.9rem;
        border-radius: 12px;
        margin-bottom: 0.7rem;
        max-width: 80%;
        font-size: 0.9rem;
        position: relative;
    }
    
    .chat-message.user {
        background: #4caf50;
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }
    
    .chat-message.assistant {
        background: #e8f5e9;
        color: #2e7d32;
        margin-right: auto;
        border-bottom-left-radius: 4px;
    }
    
    /* Clear button */
    .clear-btn {
        position: absolute;
        top: 5px;
        right: 5px;
        background: rgba(255, 255, 255, 0.3);
        border: none;
        border-radius: 50%;
        width: 20px;
        height: 20px;
        font-size: 12px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .clear-btn:hover {
        background: rgba(255, 255, 255, 0.5);
    }
    
    /* Image styling */
    .css-1v3fvcr img {
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        max-width: 300px;
        margin: 0 auto;
        display: block;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #2e7d32;
        font-size: 0.9rem;
    }
    
    .streamlit-expanderContent {
        background: #f1f8e9;
        border-radius: 0 0 6px 6px;
        padding: 0.7rem;
        font-size: 0.9rem;
    }
    
    /* Form styling */
    .stForm {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
    }
    
    /* Clear chat button */
    .clear-chat-btn {
        background: #f44336;
        color: white;
        border: none;
        padding: 0.4rem 0.8rem;
        border-radius: 4px;
        font-size: 0.8rem;
        margin-top: 0.5rem;
        cursor: pointer;
    }
    
    .clear-chat-btn:hover {
        background: #d32f2f;
    }
    </style>
""", unsafe_allow_html=True)

# Simple title
st.markdown('<div class="main-title"> Plant Disease Classifier</div>', unsafe_allow_html=True)

# Initialize session states
if 'disease_context' not in st.session_state:
    st.session_state.disease_context = ""
if 'plant_history' not in st.session_state:
    st.session_state.plant_history = []
if 'show_visualization' not in st.session_state:
    st.session_state.show_visualization = False
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Create two columns
col1, col2 = st.columns([1, 1])

# Image classification section
with col1:
    st.markdown('<div class="section-header"> PLANT DISEASE DETECTION</div>', unsafe_allow_html=True)
    
    # File uploader
    uploaded_image = st.file_uploader("Upload a plant leaf image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        # Store the uploaded image in session state
        st.session_state.uploaded_image = uploaded_image
        image = Image.open(uploaded_image)
        
        # Display medium-sized image
        st.image(image, caption="Uploaded Image", width=300)
        
        if st.button("Analyze Plant Health"):
            with st.spinner("Analyzing the image..."):
                prediction, confidence, preprocessed_img, original_img = predict_image_class(model, uploaded_image, class_indices)
                severity, percentage = assess_disease_severity(uploaded_image)
                
                # Record in history
                history_entry = {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "prediction": prediction,
                    "confidence": confidence,
                    "severity": severity,
                    "percentage": percentage
                }
                st.session_state.plant_history.append(history_entry)
                
                st.session_state.disease_context = f"{prediction} ({severity} severity - {percentage:.1f}% affected)"
                st.success(f"**Diagnosis:** {prediction}")
                st.info(f" **Severity:** {severity} ({percentage:.1f}% of leaf affected)")
                
                # Show visualization option
                st.session_state.show_visualization = True
        
        # Show visualization if available
        if st.session_state.show_visualization and st.checkbox(" Show detailed analysis", True):
            with st.spinner("Identifying diseased areas..."):
                try:
                    if st.session_state.uploaded_image is not None:
                        viz_img, disease_percentage = highlight_diseased_areas(st.session_state.uploaded_image)
                        st.image(viz_img, caption=f"Diseased areas highlighted in red ({disease_percentage:.1f}% of leaf)", width=300)
                    else:
                        st.warning("Please analyze an image first to generate visualization.")
                except Exception as e:
                    st.warning(f"Visualization could not be generated: {str(e)}")
        
        # Show history
        with st.expander("View Diagnosis History"):
            if st.session_state.plant_history:
                for i, entry in enumerate(st.session_state.plant_history):
                    st.markdown(f"""
                    <div class="card">
                        <strong>{entry['date']}:</strong> {entry['prediction']}<br>
                        <em>Confidence: {entry['confidence']:.1f}%, Severity: {entry['severity']}</em>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No diagnosis history yet.")

# Chatbot section
# Chatbot section
with col2:
    st.markdown('<div class="section-header">PLANT EXPERT CHAT</div>', unsafe_allow_html=True)
    
    # Create a container for chat messages
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages without clear buttons
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                st.markdown(f'''
                <div class="chat-message user">
                    {message["content"]}
                ''', unsafe_allow_html=True)
            else:
                # Clean up any malformed HTML in the assistant response more thoroughly
                cleaned_content = message["content"]
                # Remove common HTML tag issues
                cleaned_content = cleaned_content.replace('</div>', '')
                cleaned_content = cleaned_content.replace('<button', '')
                cleaned_content = cleaned_content.replace('</button>', '')
                cleaned_content = cleaned_content.replace('onClick=', '')
                cleaned_content = cleaned_content.replace('removeMessage', '')
                # Remove any remaining HTML tags that might be malformed
                import re
                cleaned_content = re.sub(r'<[^>]*>', '', cleaned_content)
                
                st.markdown(f'''
                <div class="chat-message assistant">
                    {cleaned_content}
                ''', unsafe_allow_html=True)
    
    # Create a form for chat input
    with st.form(key='chat_form', clear_on_submit=True):
        user_input = st.text_input("Ask about plant diseases, prevention, or treatment...", key="user_input")
        submit_button = st.form_submit_button(label='Send')
    
    # Clear all chat button
    if st.button("Clear All Chat", type="secondary"):
        st.session_state.messages = []
        st.rerun()
    
    # Process the form submission
    if submit_button and user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get response from chatbot
        with st.spinner("Thinking..."):
            response = ask_plant_disease_chatbot(user_input, st.session_state.disease_context)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Rerun to update the chat display
        st.rerun()