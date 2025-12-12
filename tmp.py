import numpy as np
import cv2
import random

# Re-using load and detect from previous step for standalone execution
def load_and_match():
    img1 = cv2.imread('input/img1.ppm')
    img2 = cv2.imread('input/img2.ppm')
    
    # Load Ground Truth H
    with open('input/H1to2p', 'r') as f:
        h_data = [[float(num) for num in line.split()] for line in f]
    H_gt = np.array(h_data)

    # Detect & Match (Using SIFT for stability)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = sift.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)
    
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Extract point correspondences
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    return img1, img2, src_pts, dst_pts, H_gt

# ------------------------------------------------------------------------
# (c) HOMOGRAPHY & RANSAC
# ------------------------------------------------------------------------

def calculateHomography(src, dst):
    """
    Computes Homography H such that dst ~= H @ src using DLT.
    src, dst: Nx2 matrices of (x, y) coordinates.
    """
    A = []
    for i in range(len(src)):
        x, y = src[i][0], src[i][1]
        xp, yp = dst[i][0], dst[i][1]
        # DLT equations
        A.append([-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x*yp, y*yp, yp])
    
    A = np.array(A)
    # SVD to find h (eigenvector corresponding to smallest eigenvalue)
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    
    # Normalize H so the last element is 1 (standard convention)
    return H / H[2, 2]

def geometricDistance(src_pt, dst_pt, H):
    """
    Calculates the reprojection error for a single point pair.
    """
    # Make src homogeneous (x, y, 1)
    p = np.array([src_pt[0], src_pt[1], 1])
    
    # Project
    p_prime = np.dot(H, p)
    
    # Normalize (convert back from homogeneous)
    p_prime = p_prime / p_prime[2]
    
    # Calculate distance to actual dst point
    return np.linalg.norm(dst_pt - p_prime[:2])

def ransac(src_pts, dst_pts, thresh=5.0, max_iters=2000):
    """
    RANSAC implementation to find best Homography.
    """
    best_H = None
    max_inliers = 0
    
    # Total number of correspondences
    n_points = len(src_pts)
    
    print(f"Starting RANSAC with {n_points} matches...")
    
    for i in range(max_iters):
        # 1. Select 4 random point correspondences
        idxs = random.sample(range(n_points), 4)
        src_sample = src_pts[idxs]
        dst_sample = dst_pts[idxs]
        
        # 2. Compute Homography for this subset
        try:
            H = calculateHomography(src_sample, dst_sample)
        except np.linalg.LinAlgError:
            continue
            
        # 3. Count inliers
        inliers_count = 0
        for j in range(n_points):
            dist = geometricDistance(src_pts[j], dst_pts[j], H)
            if dist < thresh:
                inliers_count += 1
        
        # 4. Keep best H
        if inliers_count > max_inliers:
            max_inliers = inliers_count
            best_H = H
            
    print(f"RANSAC finished. Max inliers: {max_inliers}/{n_points}")
    return best_H

# ------------------------------------------------------------------------
# (d) WARP & RESIDUAL
# ------------------------------------------------------------------------

def warp(img_src, img_dest, H):
    """
    Warps img_src to match img_dest using H.
    Displays and saves the residual.
    """
    h, w = img_dest.shape[:2]
    
    # Warp src image to destination coordinate system
    warped_img = cv2.warpPerspective(img_src, H, (w, h))
    
    # Calculate Residual (Absolute Difference)
    # Convert to grayscale for easier subtraction visual
    gray_warped = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    gray_dest = cv2.cvtColor(img_dest, cv2.COLOR_BGR2GRAY)
    
    residual = cv2.absdiff(gray_warped, gray_dest)
    
    cv2.imwrite('output/warped_img1.png', warped_img)
    cv2.imwrite('output/residual.png', residual)
    print("Saved 'warped_img1.png' and 'residual.png'")
    
    return warped_img

# ------------------------------------------------------------------------
# (e) STITCHING
# ------------------------------------------------------------------------

def get_stitched_image(img1, img2, H):
    """
    Stitches img1 and img2.
    Note: Ideally, we calculate the bounding box of the final canvas.
    For this specific pair, img1 transforms 'into' img2's plane.
    """
    # Get dimensions
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # 1. Find the size of the canvas
    # Project corners of img1 to see where they land
    corners_1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners_1_transformed = cv2.perspectiveTransform(corners_1, H)
    
    # Corners of img2 are just (0,0), (w2,0)...
    corners_2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    
    all_corners = np.concatenate((corners_1_transformed, corners_2), axis=0)
    
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    # Translation matrix to shift the image to positive coordinates if xmin/ymin < 0
    translation_dist = [-xmin, -ymin]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
    
    # Warping img1 with combined transform
    full_transform = H_translation.dot(H)
    output_shape = (xmax - xmin, ymax - ymin)
    
    stitched_img = cv2.warpPerspective(img1, full_transform, output_shape)
    
    # Place img2 onto the stitched canvas
    # We need to translate img2 as well using the same offset
    # But since img2 is the "base" (identity), we just copy it to the offset location
    sy = translation_dist[1]
    sx = translation_dist[0]
    
    # Simple overlay (img2 on top of warped img1)
    # To blend nicer, you could use masks, but overlay is sufficient for basic stitching
    stitched_img[sy:sy+h2, sx:sx+w2] = img2
    
    cv2.imwrite('output/stitched.png', stitched_img)
    print("Saved 'stitched.png'")
    
    return stitched_img

# ------------------------------------------------------------------------
# EXECUTION
# ------------------------------------------------------------------------
if __name__ == "__main__":
    img1, img2, src_pts, dst_pts, H_gt = load_and_match()
    
    # 1. Run RANSAC to calculate H
    H_calculated = ransac(src_pts, dst_pts)
    
    print("\n--- Matrix Comparison ---")
    print("Ground Truth H:\n", H_gt)
    print("RANSAC Calculated H:\n", H_calculated)
    
    # 2. Warp and Residual
    warp(img1, img2, H_calculated)
    
    # 3. Stitch
    get_stitched_image(img1, img2, H_calculated)