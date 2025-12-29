import cv2
import numpy as np
import glob
import os

def main():
    # --- Configuration ---
    input_dir = 'input'
    output_dir = 'output'
    square_size = 30  # 30mm per square
    
    # HARDCODED PATTERN SIZE based on your diagnostic run
    pattern_size = (12, 12) 
    print(f"--- Configuration ---")
    print(f"Target Pattern Size: {pattern_size}")
    print(f"Square Size: {square_size} mm")

    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get list of images and Sort them numerically (Image1, Image2, ... Image10)
    # This is crucial so "Image1" is actually index 0
    image_files = sorted(glob.glob(os.path.join(input_dir, '*.tif')), 
                         key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))

    if not image_files:
        print("Error: No images found in /input/ directory.")
        return

    print(f"Found {len(image_files)} images.")

    # --- 1. Prepare Object Points ---
    # (0,0,0), (1,0,0), (2,0,0) ... (11,11,0)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp = objp * square_size

    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane
    
    # Store success status to map image index to detected corners
    valid_images_map = {} 

    print("\n--- Step 1: Detecting Corners ---")
    
    for idx, fname in enumerate(image_files):
        img = cv2.imread(fname)
        if img is None:
            print(f"Error reading {fname}")
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Robust Detection Strategy:
        # 1. Try standard gray
        # 2. Try Histogram Equalized (for low contrast/dark images)
        images_to_try = [gray, cv2.equalizeHist(gray)]
        
        # Flags to help detection
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        
        found = False
        corners_found = None
        
        for img_ver in images_to_try:
            ret, corners = cv2.findChessboardCorners(img_ver, pattern_size, flags)
            if ret:
                found = True
                corners_found = corners
                break
        
        if found:
            print(f"[OK] {os.path.basename(fname)}")
            
            # Refine corner accuracy on the ORIGINAL gray image
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners_found, (11, 11), (-1, -1), criteria)
            
            objpoints.append(objp)
            imgpoints.append(corners2)
            valid_images_map[idx] = corners2
            
            # Optional: Save corner visualization
            # cv2.drawChessboardCorners(img, pattern_size, corners2, found)
            # cv2.imwrite(os.path.join(output_dir, f"corners_{os.path.basename(fname)}"), img)
        else:
            print(f"[FAIL] {os.path.basename(fname)} - Pattern not found.")

    if len(objpoints) < 2:
        print("\nError: Not enough good images for calibration. Need at least 2.")
        return

    # --- 2. Camera Calibration ---
    print("\n--- Step 2: Running Calibration ---")
    ret_calib, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print(f"RMS Reprojection Error: {ret_calib:.4f} pixels")
    print("\n(b) Intrinsic Matrix (K):")
    print(mtx)
    print("\n(a) Distortion Coefficients (k1, k2, p1, p2, k3):")
    print(dist.ravel())

    # --- 3. Undistort Images ---
    print("\n--- Step 3: Undistorting Images ---")
    for fname in image_files:
        img = cv2.imread(fname)
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        
        # Undistort
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        
        # Save
        out_name = os.path.join(output_dir, "undistorted_" + os.path.basename(fname))
        cv2.imwrite(out_name, dst)
    print(f"Undistorted images saved to {output_dir}/")

    # --- 4. Projection Matrices ---
    print("\n--- Step 4: Projection Matrices ---")
    
    # We need to map the list index from 'rvecs' back to the original file index
    # rvecs/tvecs corresponds to valid_images_map entries in order
    
    poses_by_file_idx = {} # Map original file index -> (R, t)

    for i, (file_idx, _) in enumerate(valid_images_map.items()):
        rvec = rvecs[i]
        tvec = tvecs[i]
        
        R, _ = cv2.Rodrigues(rvec)
        poses_by_file_idx[file_idx] = (R, tvec)
        
        # P = K [R|t]
        Rt = np.hstack((R, tvec))
        P = np.dot(mtx, Rt)
        
        print(f"P Matrix for {os.path.basename(image_files[file_idx])}:")
        print(P)
        print("-" * 20)

    # --- 5. Essential Matrix & Relative Pose ---
    print("\n--- Step 5: Essential Matrix (Ref: Image 1) ---")
    
    # Check if Image 1 (index 0) was successfully detected
    if 0 not in poses_by_file_idx:
        print("Error: Image 1 was not detected, so we cannot compute E relative to it.")
        print("Please ensure Image 1 contains the full 12x12 pattern.")
        return

    R_ref, t_ref = poses_by_file_idx[0] # Reference Pose (World -> Camera 1)
    
    for file_idx in valid_images_map.keys():
        if file_idx == 0: continue 
        
        R_curr, t_curr = poses_by_file_idx[file_idx] # Pose (World -> Camera i)

        # Calculate Relative Pose (Camera 1 -> Camera i)
        # R_rel = R_curr * R_ref^T
        # t_rel = t_curr - R_rel * t_ref
        R_rel = np.dot(R_curr, R_ref.T)
        t_rel = t_curr - np.dot(R_rel, t_ref)
        
        # Essential Matrix E = [t_rel]_x * R_rel
        t_skew = np.array([
            [0, -t_rel[2, 0], t_rel[1, 0]],
            [t_rel[2, 0], 0, -t_rel[0, 0]],
            [-t_rel[1, 0], t_rel[0, 0], 0]
        ])
        E = np.dot(t_skew, R_rel)
        
        target_name = os.path.basename(image_files[file_idx])
        print(f"\nResults for Image 1 vs {target_name}:")
        print("(e) Relative Rotation R:")
        print(R_rel)
        print("(e) Relative Translation t:")
        print(t_rel.ravel())
        print("(d) Essential Matrix E:")
        print(E)

if __name__ == "__main__":
    main()