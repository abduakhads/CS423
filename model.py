import os
import re
import json
import pandas as pd
import cv2
import pytesseract
import easyocr
import time
import numpy as np


class OCREngine:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=True) 

    def run_classical(self, binary_img):
        start = time.time()
        text = pytesseract.image_to_string(binary_img, config='--oem 0 --psm 4')
        return text, (time.time() - start) * 1000

    def run_lstm(self, raw_img):
        start = time.time()
        text = pytesseract.image_to_string(raw_img, config='--oem 1 --psm 4')
        return text, (time.time() - start) * 1000

    def run_easyocr(self, raw_img):
        start = time.time()
        # paragraph=False is critical for line-by-line logic
        result = self.reader.readtext(raw_img, detail=0, paragraph=False)
        text = "\n".join(result)
        return text, (time.time() - start) * 1000
    

# ...existing code...

if __name__ == "__main__":
    # Initialize OCR engine
    engine = OCREngine()
    
    # Load image (replace with your image path)
    image_path = "SROIE2019/test/img/X00016469671.jpg"
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        print(f"Current directory: {os.getcwd()}")
        exit(1)
    
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        exit(1)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Convert to binary image (threshold)
    _, binary_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Run classical OCR
    text, elapsed_ms = engine.run_classical(binary_img)
    
    print(f"OCR Text:\n{text}")
    print(f"\nTime taken: {elapsed_ms:.2f} ms")