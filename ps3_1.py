import numpy as np
import cv2
import sys
import random
METHODS = ['SIFT', 'MSDDetector', 'ORB'] 

#NOTE: need to install opencv and opencv-contrib libraries using pip install
#pip install opencv-python
#pip install opencv-contrib-python

# Read the images
img1 = cv2.imread('input/img1.ppm')
img2 = cv2.imread('input/img2.ppm')

cv2.imwrite("input/img1.png", img1)
cv2.imwrite("input/img2.png", img2)
# Convert to grayscale
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# loading Homography provided
h_mat = None
try:
    with open('input/H1to2p', 'r') as f:
        h_data = [[float(num) for num in line.split()] for line in f]
    h_mat = np.array(h_data)
except FileNotFoundError:
    print("Warning: H1to2p file not found. Repeatability check will be skipped.")

# Variable to store the best matches (SIFT) for the Geometry parts later
sift_data_for_geometry = None 


for method in METHODS:
    print(f"\nPROCESSING METHOD - {method}")
    
    keypoints1, descriptors1 = None, None
    keypoints2, descriptors2 = None, None
    norm_type = cv2.NORM_L2 # Default
    
    if method == 'SIFT':
        # Create a SIFT object
        sift = cv2.SIFT_create()

        # Get the keypoints and the descriptors
        keypoints1, descriptors1 = sift.detectAndCompute(img1_gray, None)
        keypoints2, descriptors2 = sift.detectAndCompute(img2_gray, None)
        norm_type = cv2.NORM_L2
        
    elif method == 'ORB':
        orb = cv2.ORB_create(nfeatures=2000)
        keypoints1, descriptors1 = orb.detectAndCompute(img1_gray, None)
        keypoints2, descriptors2 = orb.detectAndCompute(img2_gray, None)
        norm_type = cv2.NORM_HAMMING
        
    elif method == 'MSDDetector':
        print(" - Detecting MSD Keypoints...")
        msd = cv2.xfeatures2d.MSDDetector_create()
        raw_kp1 = msd.detect(img1, None)
        raw_kp2 = msd.detect(img2, None)
        
        keypoints1 = [cv2.KeyPoint(x=k.pt[0], y=k.pt[1], size=20.0) for k in raw_kp1]
        keypoints2 = [cv2.KeyPoint(x=k.pt[0], y=k.pt[1], size=20.0) for k in raw_kp2]
        
        # using SIFT instead of ORB for descriptor extraction
        sift = cv2.SIFT_create()

        keypoints1, descriptors1 = sift.compute(img1_gray, keypoints1)
        keypoints2, descriptors2 = sift.compute(img2_gray, keypoints2)
        norm_type = cv2.NORM_L2

    print(f'Descriptor 1 count: {len(descriptors1)}')
    print(f'Descriptor 2 count: {len(descriptors2)}')


    bf = cv2.BFMatcher(norm_type, crossCheck=False)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    #(a): Best Matches
    bestMatches = []
    for m_pair in matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < 0.75 * n.distance:
                bestMatches.append(m)
        elif len(m_pair) == 1:
            bestMatches.append(m_pair[0])
    
    bestMatches = sorted(bestMatches, key=lambda x: x.distance)
    print(f"Best Matches found: {len(bestMatches)}")

    out_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, bestMatches[:50], None, flags=2)
    cv2.imwrite(f"output/Matches_{method}.png", out_img)

    # (b): Repeatability
    repeatability = 0.0
    if h_mat is not None and len(bestMatches) > 0:
        pts1 = np.float32([keypoints1[m.queryIdx].pt for m in bestMatches]).reshape(-1, 1, 2)
        pts2 = np.float32([keypoints2[m.trainIdx].pt for m in bestMatches]).reshape(-1, 1, 2)
        
        # Project points
        pts1_proj = cv2.perspectiveTransform(pts1, h_mat)
        
        # Calculate error
        error = np.linalg.norm(pts1_proj - pts2, axis=2)
        correct_matches = np.sum(error < 3.0) # 3 pixel threshold
        
        avg_kps = (len(keypoints1) + len(keypoints2)) / 2.0
        repeatability = correct_matches / avg_kps
        
    print(f"Repeatability Score: {repeatability:.5f}")
    
    # Store SIFT data for the Geometry section below
    if method == 'SIFT':
        sift_data_for_geometry = (bestMatches, keypoints1, keypoints2)




# (c): Homography Estimation & RANSAC

matches_geo, kp1_geo, kp2_geo = sift_data_for_geometry
print("\n\nGeometry estimation RANSAC")
print("SIFT is being used")

def calculateHomography(src, dst):
    A = []
    for i in range(len(src)):
        x, y = src[i][0], src[i][1]
        xp, yp = dst[i][0], dst[i][1]
        A.append([-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x*yp, y*yp, yp])
    U, S, Vt = np.linalg.svd(np.array(A))
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]

def geometricDistance(src_pt, dst_pt, h):
    p = np.array([src_pt[0], src_pt[1], 1])
    p_prime = np.dot(h, p)
    if p_prime[2] != 0: p_prime /= p_prime[2]
    return np.linalg.norm(dst_pt - p_prime[:2])

def ransac(matches_list, kps1, kps2, thresh=5.0):
    src_pts = np.float32([kps1[m.queryIdx].pt for m in matches_list])
    dst_pts = np.float32([kps2[m.trainIdx].pt for m in matches_list])
    n_points = len(src_pts)
    best_H = None
    max_inliers = 0
    
    for _ in range(1000): # 1000 iterations
        if n_points < 4: break
        idxs = random.sample(range(n_points), 4)
        try:
            H_curr = calculateHomography(src_pts[idxs], dst_pts[idxs])
            inliers = 0
            # Simple inlier count
            for i in range(n_points):
                if geometricDistance(src_pts[i], dst_pts[i], H_curr) < thresh:
                    inliers += 1
            if inliers > max_inliers:
                max_inliers = inliers
                best_H = H_curr
        except: continue
    return best_H

finalH = ransac(matches_geo, kp1_geo, kp2_geo)

print("\nGiven Ground Truth H:\n", h_mat)
print("\nCalculated RANSAC H:\n", finalH)


# (d): Warp & Residual

def warp(img_src, img_dst, H):
    h, w = img_dst.shape[:2]
    warped = cv2.warpPerspective(img_src, H, (w, h))
    gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray_dst = cv2.cvtColor(img_dst, cv2.COLOR_BGR2GRAY)
    res = cv2.absdiff(gray_warped, gray_dst)
    return warped, res

if finalH is not None:
    warped_img, residual = warp(img1, img2, finalH)
    cv2.imwrite("output/Residual.png", residual)
    cv2.imwrite("output/Warped_Img1.png", warped_img)

# (e): Image Stitching

def get_stitched_image(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    corners_1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners_1_trans = cv2.perspectiveTransform(corners_1, H)
    corners_2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    
    all_corners = np.concatenate((corners_1_trans, corners_2), axis=0)
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    t_dist = [-xmin, -ymin]
    H_t = np.array([[1, 0, t_dist[0]], [0, 1, t_dist[1]], [0, 0, 1]])
    final_H = H_t.dot(H)
    output_size = (xmax - xmin, ymax - ymin)
    
    stitched = cv2.warpPerspective(img1, final_H, output_size)
    stitched[t_dist[1]:t_dist[1]+h2, t_dist[0]:t_dist[0]+w2] = img2
    return stitched

if finalH is not None:
    result = get_stitched_image(img1, img2, finalH)
    cv2.imwrite('output/stitched.png', result)
    
    cv2.imshow('Stitched Result', result)
    cv2.imshow('Residual', residual)
    print("Press any key in the image window to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()