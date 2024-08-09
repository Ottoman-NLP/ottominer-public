import cv2
import numpy as np
from PIL import Image
from pathlib import Path

rd = Path(__file__).resolve().parents[1]
fp = rd / "output" / "page_1_right_processed.png"

def fill_missing_parts(image):
    image_array = np.array(image)
    
    # Ensure the image has 3 channels (RGB) before converting to grayscale
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array  # Assume the image is already in grayscale if it doesn't have 3 channels
    
    # Denoising with Non-Local Means
    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
    
    # Local histogram equalization for improved contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equalized = clahe.apply(denoised)
    
    # Sharpening the image
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(equalized, -1, kernel)
    
    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 15)
    
    # Morphological operations to fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=15)
    
    # Additional step: morphological opening to refine the image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=4)
    
    # Dilation to fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    dilated = cv2.dilate(opened, kernel, iterations=4)
    
    return Image.fromarray(dilated)

# Use the function on your image
image_path = fp
image = Image.open(image_path)
processed_image = fill_missing_parts(image)
processed_image.show()
processed_image.save("processed_image_filled.png")
