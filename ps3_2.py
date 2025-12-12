import numpy as np
import cv2
import sys


img1 = cv2.imread('input/img1.ppm')
img2 = cv2.imread('input/img2.ppm')


img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


# (a): Dense Optical Flow

def dense_optical_flow(prev_gray, next_gray, prev_img):
    # I have tuned some parameters for smoother results after first bad results:
    # pyr_scale=0.5: Classic pyramid scale
    # levels=3: Number of layers
    # winsize=30: (increase) larger window averages out noise better
    # iterations=3: steps per level
    # poly_n=7: (increase) size of pixel neighborhood. first tried 5, but 7 is smoother.
    # poly_sigma=1.5: (increase) standard deviation for Gaussian. first tried 1.2, but 1.5 gave better results.
    flow = cv2.calcOpticalFlowFarneback(next_gray, prev_gray, None, 
                                        pyr_scale=0.5, 
                                        levels=3, 
                                        winsize=30,
                                        iterations=3, 
                                        poly_n=7,
                                        poly_sigma=1.5, 
                                        flags=0)
    
    # warping img1 to match img2 using the flow
    h, w = prev_gray.shape
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
    
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]
    
    # Remap
    warped_img = cv2.remap(prev_img, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    
    return warped_img


warped_dense = dense_optical_flow(img1_gray, img2_gray, img1)

residual_dense = cv2.absdiff(cv2.cvtColor(warped_dense, cv2.COLOR_BGR2GRAY), img2_gray)

cv2.imwrite("output/optical_warp.png", warped_dense)
cv2.imwrite("output/optical_residual.png", residual_dense)

cv2.imshow("Dense Flow Warp", warped_dense)
cv2.imshow("Dense Flow Residual", residual_dense)






# (b): Sparse Tracking"

# Detect Shi-Tomasi Corners
# inceased minDistance slightly to avoid clustering
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=10, 
                      blockSize=7)

p0 = cv2.goodFeaturesToTrack(img1_gray, mask=None, **feature_params)

# Lucas-Kanade Optical Flow
# Increased window size here as well for consistency
lk_params = dict(winSize=(21, 21), # <- Increased from 15x15
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

p1, st, err = cv2.calcOpticalFlowPyrLK(img1_gray, img2_gray, p0, None, **lk_params)

good_new = p1[st == 1]
good_old = p0[st == 1]

mask = np.zeros_like(img1)
output_lk = img2.copy()
color = np.random.randint(0, 255, (100, 3))

for i, (new, old) in enumerate(zip(good_new, good_old)):
    a, b = new.ravel()
    c, d = old.ravel()
    mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
    output_lk = cv2.circle(output_lk, (int(a), int(b)), 5, color[i].tolist(), -1)

final_lk_img = cv2.add(output_lk, mask)

cv2.imwrite("output/lucas_kanade_tracking.png", final_lk_img)

cv2.imshow("Sparse Flow Tracks", final_lk_img)

print("\nPress any key in the image windows to exit...")
cv2.waitKey(0)
cv2.destroyAllWindows()