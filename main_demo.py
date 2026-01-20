import cv2
import os
from action_forge import create_forgery
from detection_model import analyze_image, build_paper_model

def main():
    print("--- DIGITAL FORENSICS: RESAMPLING DETECTION PROJECT ---")
    
    # Step 1: Input Check
    image_name = 'input_image.jpg' 
    if not os.path.exists(image_name):
        print(f"ERROR: Please place an image named '{image_name}' in the folder.")
        return

    # Step 2: The "Action" (Forgery)
    # We rotate the image, which creates pixel resampling artifacts
    print("\n[STEP 1] Performing Action: Generating Forged Image...")
    forged_name = 'forged_image.jpg'
    create_forgery(image_name, forged_name, mode='rotate')

    # Step 3: The "Detection" (Analysis)
    # We use the method from the paper (Laplacian Filter + AI Model) to spot the artifacts
    print("\n[STEP 2] Performing Analysis: Detecting Resampling Artifacts...")
    
    # Load our AI Model structure 
    model = build_paper_model()
    print("AI Model Loaded successfully.")
    
    # Visualize the detection (Heatmap)
    print("Generating Heatmap based on Paper's method...")
    analyze_image(forged_name, model)

if __name__ == "__main__":
    main()