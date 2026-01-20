import cv2
import numpy as np

def create_forgery(image_path, output_path, mode='rotate'):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from '{image_path}'. Make sure the file exists.")
        return

    rows, cols, _ = img.shape
    
    if mode == 'rotate':
        # Rotate by 15 degrees (This creates the resampling artifacts)
        # We rotate around the center
        M = cv2.getRotationMatrix2D((cols/2, rows/2), 15, 1)
        forged_img = cv2.warpAffine(img, M, (cols, rows))
        print(f"Applied Rotation to {image_path}")

    elif mode == 'resize':
        # Upscale the image (This also creates resampling artifacts)
        forged_img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        print(f"Applied Resizing to {image_path}")
        
    else:
        print("Unknown mode")
        return

    # Save the forged image
    cv2.imwrite(output_path, forged_img)
    print(f"Forgery saved successfully as: {output_path}")

# --- TEST IT ---
if __name__ == "__main__":
    # This block only runs if you run this file directly
    create_forgery('input_image.jpg', 'forged_image.jpg', mode='rotate')