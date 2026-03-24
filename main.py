import os
import re
import json
import pandas as pd
import cv2
import pytesseract
import easyocr
import time
import numpy as np
import warnings

warnings.filterwarnings("ignore", message=".*pin_memory.*")


def clean_currency(value):
    """
    Standardizes currency strings.
    Input: "$1,200.50" -> Output: "1200.50"
    """
    if not isinstance(value, str):
        return "0.00"
    
    # Remove everything that is NOT a digit or a dot
    cleaned = re.sub(r'[^\d.]', '', value)
    
    try:
        return "{:.2f}".format(float(cleaned))
    except ValueError:
        return "0.00"


# Preprocessor
class Preprocessor:
    def process(self, image):
        """
        Returns two versions of the image:
        1. Gray: For LSTM (Preserves soft edges/gradients)
        2. Binary (Otsu): For Classical (High contrast, solid shapes)
        """
        # 1. Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 2. Otsu's Thresholding (CRITICAL for Tesseract Legacy)
        # We used Otsu because SROIE images are high-contrast scans.
        # Adaptive thresholding introduced noise that breaks the legacy engine.
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return gray, binary


# OCR Engine
class OCREngine:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=True) 

    def run_classical(self, binary_img):
        start = time.time()
        # --oem 0: Legacy Engine
        text = pytesseract.image_to_string(binary_img, config='--oem 0 --psm 6')
        return text, (time.time() - start) * 1000

    def run_lstm(self, raw_img):
        start = time.time()
        # --oem 1: Neural Network Engine (LSTM)
        text = pytesseract.image_to_string(raw_img, config='--oem 1 --psm 4')
        return text, (time.time() - start) * 1000

    def run_easyocr(self, raw_img):
        start = time.time()
        # paragraph=False: Keep structure line-by-line for keyword searching
        result = self.reader.readtext(raw_img, detail=0, paragraph=False) 
        text = "\n".join(result)
        return text, (time.time() - start) * 1000


def get_ground_truth(filename, entities_folder):
    """ Reads the JSON Ground Truth from SROIE """
    txt_filename = filename.replace('.jpg', '.txt')
    path = os.path.join(entities_folder, txt_filename)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return clean_currency(data.get("total", "0.00"))
    except Exception as e:
        return "Unknown"


def extract_linear_total(text):
    """
    OPTIMIZED FOR TESSERACT (LSTM & CLASSICAL)
    Strictly looks for the price on the SAME LINE as keywords.
    """
    lines = text.split('\n')
    price_pattern = r'(\d{1,3}(?:,\d{3})*\.\d{2})'
    
    target_keywords = ["total", "amount", "balance", "payable", "nett", "grand", "rm", "subtotal"]
    forbidden_keywords = ["cash", "change", "tender", "received", "tax", "gst", "qty"]
    
    candidates = []

    for line in lines:
        line_lower = line.lower()
        
        # filtering Garbage
        if any(bad in line_lower for bad in forbidden_keywords):
            continue

        # must contain a number
        matches = re.findall(price_pattern, line)
        if not matches:
            continue

        # analyzing the number
        val_str = matches[-1].replace(',', '.') # taking the last number on the line
        try:
            val_float = float(val_str)
        except:
            continue
            
        # sanity check
        if val_float > 2000 or val_float < 0.10: continue

        # strict scoring
        score = 1
        if any(good in line_lower for good in target_keywords):
            score = 10 # boost if "Total" is on the same line
        
        candidates.append((val_float, score))

    if not candidates: return "0.00"
    
    # sorting by Score first, then Value
    candidates.sort(key=lambda x: (x[1], x[0]), reverse=True)
    return "{:.2f}".format(candidates[0][0])


def extract_spatial_total(text):
    """
    OPTIMIZED FOR EASYOCR
    looks ahead logic because EasyOCR splits lines into blocks.
    """
    # EasyOCR often puts spaces in numbers: "1 0 . 0 0"
    # We pre-cleaned the text to fix this
    text = re.sub(r'(\d)\s+\.\s+(\d)', r'\1.\2', text) # Fix "10 . 00" -> "10.00"
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)       # Fix "1 0" -> "10"
    
    lines = text.split('\n')
    price_pattern = r'(\d{1,3}(?:,\d{3})*\.\d{2})'
    
    target_keywords = ["total", "amount", "balance", "grand", "nett"]
    forbidden_keywords = ["cash", "change", "tender"]
    
    candidates = []
    keyword_seen_recently = False
    lines_since_keyword = 0

    for line in lines:
        line_lower = line.lower()
        
        # reset if we hit "Cash"
        if any(bad in line_lower for bad in forbidden_keywords):
            keyword_seen_recently = False
            continue

        # checking for Keyword
        if any(good in line_lower for good in target_keywords):
            keyword_seen_recently = True
            lines_since_keyword = 0
        elif keyword_seen_recently:
            lines_since_keyword += 1
            if lines_since_keyword > 3: # stop looking after 3 lines
                keyword_seen_recently = False

        # extracting Number
        matches = re.findall(price_pattern, line)
        for match in matches:
            val_str = match.replace(',', '.')
            try:
                val_float = float(val_str)
            except: continue

            if val_float > 2000 or val_float < 0.10: continue

            # scoring
            score = 1
            if keyword_seen_recently:
                score = 5 # boost if we saw "Total" recently
                
            candidates.append((val_float, score))

    if not candidates: return "0.00"
    candidates.sort(key=lambda x: (x[1], x[0]), reverse=True)
    return "{:.2f}".format(candidates[0][0])


def run_experiment():
    preprocessor = Preprocessor()
    ocr_engine = OCREngine()
    
    img_folder = "SROIE2019/test/img"
    entities_folder = "SROIE2019/test/entities" 
    
    results = []
    
    if not os.path.exists(img_folder):
        print("❌ Error: 'data/img' folder missing.")
        return

    image_files = sorted([f for f in os.listdir(img_folder) if f.endswith('.jpg')])[100:]

    print(f"🚀 Starting Final Experiment on {len(image_files)} images...")
    print(f"{'Filename':<20} | {'Truth':<8} | {'Clas':<8} | {'LSTM':<8} | {'Easy':<8}")
    print("-" * 75)

    for filename in image_files:
        img_path = os.path.join(img_folder, filename)
        truth_total = get_ground_truth(filename, entities_folder)
        
        img = cv2.imread(img_path)
        if img is None: continue
        
        gray, binary = preprocessor.process(img)
        
        # 1. Run Engines
        txt_c, time_c = ocr_engine.run_classical(binary)
        txt_l, time_l = ocr_engine.run_lstm(gray)
        txt_e, time_e = ocr_engine.run_easyocr(img) 
        
        # 2. Extract using SPECIALIZED Logic
        # Tesseract engines get the Strict Linear Logic
        pred_c = extract_linear_total(txt_c)
        pred_l = extract_linear_total(txt_l)
        
        # EasyOCR gets the Loose Spatial Logic
        pred_e = extract_spatial_total(txt_e)
        
        # 3. Compare
        icon_c = "✅" if pred_c == truth_total else "❌"
        icon_l = "✅" if pred_l == truth_total else "❌"
        icon_e = "✅" if pred_e == truth_total else "❌"
        
        print(f"{filename:<20} | {truth_total:<8} | {pred_c:<8} {icon_c} | {pred_l:<8} {icon_l} | {pred_e:<8} {icon_e}")
        
        results.append({
            "Filename": filename,
            "Ground_Truth": truth_total,
            "Classical_Pred": pred_c,
            "Classical_Time_ms": int(time_c),
            "LSTM_Pred": pred_l,
            "LSTM_Time_ms": int(time_l),
            "EasyOCR_Pred": pred_e,
            "EasyOCR_Time_ms": int(time_e),
            "Classical_Correct": (pred_c == truth_total),
            "LSTM_Correct": (pred_l == truth_total),
            "EasyOCR_Correct": (pred_e == truth_total)
        })

    df = pd.DataFrame(results)
    df.to_csv("final_results.csv", index=False)
    
    print("\n" + "="*40)
    print("📊 EXPERIMENT SUMMARY")
    print("="*40)
    print(f"Classical Accuracy: {df['Classical_Correct'].mean():.1%}")
    print(f"LSTM Accuracy:      {df['LSTM_Correct'].mean():.1%}")
    print(f"EasyOCR Accuracy:   {df['EasyOCR_Correct'].mean():.1%}")

if __name__ == "__main__":
    run_experiment()

    try:
        import plot
        plot.main()
    except ImportError:
        print("⚠️ 'plot.py' not found. Please run it separately to generate graphs.")