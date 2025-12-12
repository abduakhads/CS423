import numpy as np
import cv2
import random
from matplotlib import pyplot as plt
METHOD = 'SIFT'#'MSDDetector''ORB' 

#NOTE: need to install opencv and opencv-contrib libraries using pip install
#pip install opencv-python
#pip install opencv-contrib-python

# Read the images
img1 = cv2.imread('input/img1.ppm')
img2 = cv2.imread('input/img2.ppm')

cv2.imwrite("input/img1.png",img1)
cv2.imwrite("input/img2.png",img2)
# Convert the images to grayscale
img1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)


if METHOD == 'SIFT':
    
    print('Calculating SIFT features...\n')
    
    
    # Create a SIFT object
    sift = cv2.SIFT_create()

    # Get the keypoints and the descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(img1_gray,None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2_gray,None)
    # keypoints object includes position, size, angle, etc.
    # descriptors is an array. For sift, each row is a 128-length feature vector
    
    print('Descriptor 1 : ',len(descriptors1))
    print('Descriptor 2 : ',len(descriptors2))
    
   
elif METHOD == 'MSDDetector':
    print('Calculating MSDDetector keypoints and ORB features...\n')
    
    # Create a SURF object
    msd = cv2.xfeatures2d.MSDDetector_create()
    
    keypoints1 = msd.detect(img1, None)
    keypoints2 = msd.detect(img2, None)

    # 3) Use a separate descriptor extractor, e.g., ORB
    orb = cv2.ORB_create()

    # Compute descriptors at MSD keypoint locations    
    keypoints1, descriptors1 = orb.compute(img1_gray,keypoints1)
    keypoints2, descriptors2 = orb.compute(img2_gray,keypoints2)
    print('Descriptor 1 : ',len(descriptors1))
    print('Descriptor 2 : ',len(descriptors2))

elif METHOD == 'ORB':
    print('Calculating ORB features...\n')

    # Create a ORB object
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img1_gray,None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2_gray,None)
    print('Descriptor 1 : ',len(descriptors1))
    print('Descriptor 2 :',len(descriptors2))
    # Note: Try cv2.NORM_HAMMING for this feature
    

# Draw the keypoints
img1_key = np.zeros((img1.shape))  
img1_key = cv2.drawKeypoints(image=img1, outImage=img1_key, keypoints=keypoints1, 
                        flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, 
                        color = (0, 0, 255))
img2_key= np.zeros((img2.shape)) 
img2_key = cv2.drawKeypoints(image=img2, outImage=img2_key, keypoints=keypoints2, 
                        flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, 
                        color = (0, 0, 255))



# Display the images
cv2.imwrite('output/SIFT Keypoints 1.png', img1_key)
cv2.imwrite('output/SIFT Keypoints 2.png', img2_key)

# Create a brute-force descriptor matcher object
if METHOD == 'ORB':
   bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
else:
   bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)



matching = bf.knnMatch(descriptors1,descriptors2,k=2)

print("# of matches between descriptors:",len(matching))

# ------------------------------------------------------------------------
#Problem-1

# Best Match 
        
#print("Best Matches",len(bestMatches))
#cv2.imwrite("output/Matches.png",best)

# ------------------------------------------------------------------------
#Repeatability

#print("Repeatability:",repeatability)

#-------------------------------------------------------------------------
#Homography Estimation

#def calculateHomography(correspondences):

#Calculate the geometric distance between estimated points and original points
#def geometricDistance(correspondence, h):

#def ransac(corr, thresh):
# print("Given homography matrix:")
# print(h_mat)
# print("Ransac homography matrix:")
# print(finalH)

#-----------------------------------------------------------------------------
#def warp(img1,img2):
    
#residual=warp(img1,img2)
#cv2.imwrite("output/Residual.png",residual)

#-----------------------------------------------------------------------------
#Image Stitching
#def get_stitched_image(img1, img2, finalH ):
  
# result = get_stitched_image(img2, img1, finalH)
# cv2.imwrite('output/stitched.png', result)


      

