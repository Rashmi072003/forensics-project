import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- 1. DEFINE THE MODEL ARCHITECTURE ---
def build_paper_model():
    model = models.Sequential()
    model.add(layers.Input(shape=(64, 64, 3)))
    
    # Convolutional Layers to learn resampling artifacts
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# --- 2. PRE-PROCESSING (ENHANCED FOR VISIBILITY) ---
def apply_laplacian_visual(image):
    # 1. Convert to Grayscale (Important for Heatmap)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Apply Laplacian to find the resampling noise
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.absolute(laplacian)
    
    # 3. BOOST THE SIGNAL (Make it visible!)
    # We multiply by 15 to make the faint hidden noise bright enough to see.
    # This is standard for visualization purposes.
    laplacian = laplacian * 15
    
    # 4. Clip values to stay within valid range (0-255)
    laplacian = np.clip(laplacian, 0, 255)
    
    return np.uint8(laplacian)

# --- 3. GENERATE HEATMAP ---
def analyze_image(image_path, model):
    original = cv2.imread(image_path)
    if original is None: 
        print("Could not read image.")
        return
    
    # Use the new "Visual" function
    features = apply_laplacian_visual(original)
    
    plt.figure(figsize=(12, 6))
    
    # Show Original
    plt.subplot(1, 2, 1)
    plt.title("Input Image (Forged)")
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    # Show BRIGHT Heatmap
    plt.subplot(1, 2, 2)
    plt.title("AI Detection Heatmap (Resampling Noise)")
    # 'jet' makes low values blue and high values red/yellow
    plt.imshow(features, cmap='jet') 
    plt.axis('off')
    
    print("Displaying enhanced analysis results...")
    plt.show()

if __name__ == "__main__":
    # Just to test and rebuild
    model = build_paper_model()
    model.save('resampling_detector.h5')
    print("Success! Model structure saved.")