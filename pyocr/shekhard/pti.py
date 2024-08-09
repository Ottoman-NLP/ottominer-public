import os
import ctypes
import numpy as np
from pathlib import Path
import sys
import fitz
from ArabicOcr import arabicocr
import torch
import cv2
import matplotlib.pyplot as plt
import logging
from glob import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    ctypes.CDLL("libiomp5md.dll", mode=ctypes.RTLD_GLOBAL)
except OSError:
    pass
try:
    ctypes.CDLL("libomp140.dll", mode=ctypes.RTLD_GLOBAL)
except OSError:
    pass

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device.upper()} for OCR processing.")

rd = Path(__file__).resolve().parents[1]
pdf_dir = rd / "output"
pdf_file = pdf_dir / "page_1_right_processed.png"

if not pdf_file.is_file():
    print(f"Error: The file {pdf_file} does not exist.")
    sys.exit(1)

output_dir = Path("output_images")
output_dir.mkdir(exist_ok=True)

doc = fitz.open(pdf_file)

for page in doc:
    pix = page.get_pixmap()
    pix.save(output_dir / f"page_{page.number + 1}.png")

image_path = output_dir / "page_1.png"
out_image = output_dir / 'out.jpg'

if not image_path.is_file():
    print(f"Error: The file {image_path} does not exist.")
    sys.exit(1)

def preprocess_image(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img.shape[1] * 3, img.shape[0] * 3), interpolation=cv2.INTER_CUBIC)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    preprocessed_image_path = str(output_dir / "pti_pro_1.png")
    cv2.imwrite(preprocessed_image_path, img)
    return preprocessed_image_path

def preprocess(image):
    if len(image.shape) == 2:
        gray_img = image
    else:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    gray_img = cv2.bitwise_not(gray_img)
    binary_img = binary_otsus(gray_img, 0)
    deskewed_img = deskew(binary_img)
    
    return deskewed_img

def binary_otsus(img, threshold):
    if threshold == 0:
        _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return binary_img

def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def projection(img, axis):
    if axis == 'horizontal':
        return np.sum(img, axis=1)
    elif axis == 'vertical':
        return np.sum(img, axis=0)

def projection_segmentation(clean_img, axis, cut=2):
    segments = []
    start = -1
    cnt = 0

    projection_bins = projection(clean_img, axis)
    for idx, projection_bin in enumerate(projection_bins):
        if projection_bin != 0:
            cnt = 0
        if projection_bin != 0 and start == -1:
            start = idx
        if projection_bin == 0 and start != -1:
            cnt += 1
            if cnt >= cut:
                if axis == 'horizontal':
                    segments.append(clean_img[max(start-1, 0):idx, :])
                elif axis == 'vertical':
                    segments.append(clean_img[:, max(start-1, 0):idx])
                cnt = 0
                start = -1
    return segments

def line_horizontal_projection(image, cut=2):
    clean_img = preprocess(image)
    lines = projection_segmentation(clean_img, axis='horizontal', cut=cut)
    return lines

def word_vertical_projection(line_image, cut=2):
    line_words = projection_segmentation(line_image, axis='vertical', cut=cut)
    line_words.reverse()
    return line_words

def extract_words(img, visual=0):
    lines = line_horizontal_projection(img)
    words = []
    for idx, line in enumerate(lines):
        if visual:
            cv2.imwrite(str(output_dir / f'line{idx}.png'), line)
        line_words = word_vertical_projection(line)
        for w in line_words:
            words.append((w, line))
    if visual:
        for idx, word in enumerate(words):
            cv2.imwrite(str(output_dir / f'word{idx}.png'), word[0])
    return words

preprocessed_image_path = preprocess_image(image_path)

preprocessed_img = cv2.imread(preprocessed_image_path, cv2.IMREAD_GRAYSCALE)
words_segments = extract_words(preprocessed_img)

results = []
for word_segment, line in words_segments:
    word_img_path = str(output_dir / "word_segment.png")
    cv2.imwrite(word_img_path, word_segment)
    result = arabicocr.arabic_ocr(word_img_path, str(out_image))
    results.extend(result)

print(results)

words = [result[1] for result in results]
with open('file.txt', 'w', encoding='utf-8') as myfile:
    myfile.write("\n".join(words))

annotations = [str(result[0]) for result in results]
with open('annotations.txt', 'w', encoding='utf-8') as myfile:
    myfile.write("\n".join(annotations))

if out_image.is_file():
    img = cv2.imread(str(out_image), cv2.IMREAD_UNCHANGED)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
else:
    print(f"Error: The file {out_image} does not exist.")
