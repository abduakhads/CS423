import cv2
import numpy as np
import sys

def find_true_pattern():
    # Load one of the problematic images
    img_path = 'input/Image3.tif'
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Could not load {img_path}. Check file path.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply enhancement to help detection
    gray_enhanced = cv2.equalizeHist(gray)
    
    print(f"Scanning {img_path} for checkerboard pattern...")
    print("Trying sizes from 3x3 to 14x14...")

    # Brute-force dimensions
    for rows in range(3, 15):
        for cols in range(3, 15):
            # Check for (cols, rows)
            pattern = (cols, rows)
            
            # Use adaptive thresholding for robustness
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
            
            ret, corners = cv2.findChessboardCorners(gray, pattern, flags)
            
            if not ret:
                # Try enhanced version
                ret, corners = cv2.findChessboardCorners(gray_enhanced, pattern, flags)
                
            if ret:
                print(f"\n[SUCCESS] Found pattern: {cols} columns x {rows} rows")
                print("Update your 'square_size' variable in the main script if needed,")
                print(f"and set: pattern_size = ({cols}, {rows})")
                
                # Draw and save to verify it's the FULL board
                cv2.drawChessboardCorners(img, pattern, corners, ret)
                cv2.imwrite('debug_pattern_result.png', img)
                print("Saved 'debug_pattern_result.png' with detected corners. Check it!")
                return

    print("\n[FAILURE] No standard pattern found. The image might be too dark, blurry, or the board is occluded.")

if __name__ == "__main__":
    find_true_pattern()