import cv2
import numpy as np
import matplotlib.pyplot as plt
import os, sys

def hough_circles_acc(img_edges, min_r, max_r):
    rows, cols = img_edges.shape
    
    # Dimensions: height x width x (max_r - min_r)
    r_range = max_r - min_r
    H = np.zeros((rows, cols, r_range), dtype=np.uint64)
    
    # y_idxs and x_idxs are arrays of coordinates where edges exist
    y_idxs, x_idxs = np.nonzero(img_edges) 
    
    # Pre-calculating sin and cos for a full circle (0 to 360 degrees)
    # a lookup table speeds up the process
    thetas = np.deg2rad(np.arange(0, 360))
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)
    
    # Voting in the accumulator
    # We iterate through possible radii
    for r_idx, r in enumerate(range(min_r, max_r)):
        # For this radius r, calculate the offset circle for ALL edge points at once
        # This is much faster than looping through every pixel
        
        # Calc circle offsets (a, b) from edge (x, y)
        # a = x - r * cos(theta)
        # b = y - r * sin(theta)
        
        # We create a grid of potential centers around the edge pixels
        for theta_idx in range(len(thetas)):
            a = np.round(x_idxs - r * cos_thetas[theta_idx]).astype(int)
            b = np.round(y_idxs - r * sin_thetas[theta_idx]).astype(int)
            
            # Check bounds to ensure we don't vote outside the image
            valid_indices = (a >= 0) & (a < cols) & (b >= 0) & (b < rows)
            
            # Increment accumulator -> We flatten the indices to use np.add.at which handles unbuffered in-place addition
            # (Just doing H[b, a, r_idx] += 1 would miss duplicates in the same step)
            valid_b = b[valid_indices]
            valid_a = a[valid_indices]
            
            # Vote
            H[valid_b, valid_a, r_idx] += 1

    return H


def hough_circles_peaks(H, min_r, t, s):
    # Thresholding by finding indices where votes > t which creates a list of potential centers
    potential_peaks_idx = np.where(H > t)
    
    # get coordinates and vote counts
    y_coords = potential_peaks_idx[0]
    x_coords = potential_peaks_idx[1]
    r_indices = potential_peaks_idx[2]
    votes = H[y_coords, x_coords, r_indices]
    
    # Sort by vote strength (descending)
    sorted_indices = np.argsort(votes)[::-1]
    
    circles = []
    
    # Select top 's' peaks
    for i in range(min(s, len(sorted_indices))):
        idx = sorted_indices[i]
        x = x_coords[idx]
        y = y_coords[idx]
        r = r_indices[idx] + min_r # min_r back to get actual radius
        circles.append((x, y, r))
        
    return circles


def draw_circles(img, circles):
    output_img = img.copy()
    for (x, y, r) in circles:
        # outer circle
        cv2.circle(output_img, (x, y), r, (0, 255, 0), 2) # green
        # center of the circle 
        cv2.circle(output_img, (x, y), 2, (0, 0, 255), 3) # red
    
    return output_img


if __name__ == "__main__":
    #  image load
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        sys.exit("Usage: python ps2_1.py <path_to_image>")

    # parameters
    min_radius = 20
    max_radius = 100
    t = 110  # Minimum votes to consider a circle
    s = 10   # Number of circles to detect


    original_img = cv2.imread(input_path)
    
    # chagning to Grayscale
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    
    # detecting edges using Canny
    edges = cv2.Canny(gray_img, 100, 200) 
    
    # running Hough Transform
    print("Computing Accumulator...")
    acc = hough_circles_acc(edges, min_radius, max_radius)
    
    # finding Peaks
    peaks = hough_circles_peaks(acc, min_radius, t, s)
    
    print(f"Found {len(peaks)} circles.")
    print("Circles (x, y, r):", peaks)
    
    # drawing Results
    result_img = draw_circles(original_img, peaks)
    
    # OpenCV implementation for comparison
    # param1: Canny edge high threshold, param2: accumulator threshold
    opencv_circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1, 50,
                                        param1=100, param2=t, minRadius=min_radius, maxRadius=max_radius)
    
    if opencv_circles is not None:
        opencv_circles = np.uint16(np.around(opencv_circles))
        opencv_res = original_img.copy()
        for i in opencv_circles[0, :]:
            # Draw the outer circle
            cv2.circle(opencv_res, (i[0], i[1]), i[2], (255, 0, 0), 2)
    
    # results
    if not os.path.exists('output'):
        os.makedirs('output')
        
    cv2.imwrite('output/my_hough.jpg', result_img)
    cv2.imwrite('output/edges.jpg', edges)
    cv2.imwrite('output/opencv_hough.jpg', opencv_res)
    
    plt.figure(figsize=(15,5))
    plt.subplot(131), plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)), plt.title('My Implementation')
    plt.subplot(132), plt.imshow(cv2.cvtColor(opencv_res, cv2.COLOR_BGR2RGB)), plt.title('OpenCV Implementation')
    plt.subplot(133), plt.imshow(edges, cmap='gray'), plt.title('Edges')
    plt.show()


















#https://www.youtube.com/watch?v=Ltqt24SQQoI https://www.youtube.com/watch?v=VfDR-8OSExk