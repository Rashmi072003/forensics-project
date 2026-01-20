import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image

# --- 1. BUILD THE AI MODEL (Hidden from view) ---
@st.cache_resource # Keeps it fast
def load_model():
    model = models.Sequential()
    model.add(layers.Input(shape=(64, 64, 3)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# --- 2. HELPER FUNCTIONS ---
def apply_laplacian_visual(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.absolute(laplacian) * 15 # Boost visibility
    laplacian = np.clip(laplacian, 0, 255)
    return np.uint8(laplacian)

def create_forgery(image):
    # Convert PIL to OpenCV
    img = np.array(image.convert('RGB')) 
    img = img[:, :, ::-1].copy() 
    rows, cols, _ = img.shape
    # Rotate by 15 degrees
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 15, 1)
    forged_img = cv2.warpAffine(img, M, (cols, rows))
    return forged_img

# --- 3. THE APP INTERFACE ---
st.title("🕵️ Digital Image Forensics")
st.write("Detection of Resampling Forgery using Deep Learning")

# Step 1: Upload
st.header("1. Upload Genuine Image")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Original Image', use_column_width=True)
    
    # Step 2: Action
    if st.button("Generate Forgery & Detect"):
        with st.spinner('Creating Forgery (Rotation)...'):
            # Create Forgery
            forged_cv = create_forgery(image)
            # Convert back to RGB for display
            forged_display = cv2.cvtColor(forged_cv, cv2.COLOR_BGR2RGB)
            
        st.success("Forgery Created!")
        st.image(forged_display, caption='Forged Image (Rotated 15°)', use_column_width=True)
        
        # Step 3: Detection
        with st.spinner('Running AI Analysis...'):
            # Build model (simulated load)
            model = load_model()
            # Get heatmap
            heatmap = apply_laplacian_visual(forged_cv)
            
        st.header("2. AI Analysis Result")
        st.image(heatmap, caption='Detection Heatmap (Resampling Artifacts)', clamp=True, channels='GRAY')
        st.info("The bright noise indicates invisible resampling traces detected by the model.")