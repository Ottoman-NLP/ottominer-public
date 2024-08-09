import os
import logging
from PIL import Image
import pytesseract
from pathlib import Path
import fitz
import cv2
import numpy as np
from skimage import restoration

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

rd = Path(__file__).resolve().parents[1]
pdf_file = rd / "pdfs" / "adalet" / "adalet-sayfa-7.pdf"
output_dir = rd / "output"
output_dir.mkdir(parents=True, exist_ok=True)

def preprocess_image(image):
    image_array = np.array(image)
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    # Denoising with Non-Local Means
    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
    
    # Local histogram equalization for improved contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equalized = clahe.apply(denoised)
    
    # Sharpening the image
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(equalized, -1, kernel)
    
    # Adaptive thresholding with a larger block size
    binary = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 15)
    
    # Morphological operations to connect nearby components
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Remove small noise
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.ones(connected.shape[:2], dtype="uint8") * 255
    for c in contours:
        if cv2.contourArea(c) < 100:
            cv2.drawContours(mask, [c], 0, 0, -1)
    cleaned = cv2.bitwise_and(connected, connected, mask=mask)
    
    return Image.fromarray(cleaned)

def capture_words(binary_image):
    # Find all connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    
    # Filter components based on size to get potential words
    min_size = 100  # Adjust this value based on your document characteristics
    word_boxes = []
    for i in range(1, num_labels):  # Start from 1 to skip background
        if stats[i, cv2.CC_STAT_AREA] > min_size:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            word_boxes.append((x, y, w, h))
    
    return word_boxes

def split_book_page(image):
    width, height = image.size
    mid = width // 2
    right_page = image.crop((0, 0, mid, height))
    left_page = image.crop((mid, 0, width, height))
    return right_page, left_page

def pdf_to_txt(pdf_path, output_txt_path, lang='osd'):
    logging.info(f"Opening PDF file: {pdf_path}")
    pdf_document = fitz.open(pdf_path)
    
    full_text = ""
    for page_num in range(len(pdf_document)):
        logging.info(f"Processing page {page_num + 1} of {len(pdf_document)}")
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Increase resolution
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        right_page, left_page = split_book_page(img)
        
        for i, sub_page in enumerate([right_page, left_page]):
            processed_img = preprocess_image(sub_page)
            processed_img.save(output_dir / f"page_{page_num + 1}_{['right', 'left'][i]}_processed.png")
            
            logging.info(f"Extracting text from page {page_num + 1} ({['right', 'left'][i]})")
            text = pytesseract.image_to_string(processed_img, lang=lang, config='--psm 6 --oem 1 -c tessedit_char_whitelist=ابتثجحخدذرزسشصضطظعغفقكلمنهوي')
            logging.info(f"Extracted text from page {page_num + 1} ({['right', 'left'][i]}): {text[:100]}")
            full_text += text + "\n\n"
            
            # Use capture_words function
            word_boxes = capture_words(np.array(processed_img))
            for j, (x, y, w, h) in enumerate(word_boxes):
                word_img = processed_img.crop((x, y, x+w, y+h))
                word_text = pytesseract.image_to_string(word_img, lang=lang, config='--psm 7 --oem 1')
                full_text += word_text + " "
    
    pdf_document.close()
    
    full_txt_file = output_dir / f"full_text_{pdf_path.stem}.txt"
    logging.info(f"Writing extracted text to {full_txt_file}")
    with open(full_txt_file, 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    logging.info("Text extraction complete")
    return full_text

extracted_text = pdf_to_txt(pdf_file, output_dir, lang='osd')
print(extracted_text)
