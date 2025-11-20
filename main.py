# import the opencv library
import cv2
import numpy as np
  
# define a video capture object
vid = cv2.VideoCapture(1)
if not vid.isOpened():
    print("❌ Cannot open camera")
    exit()

kernel_x = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]])

kernel_y = np.array([[-1, -1, -1],
                   [0, 0, 0],
                   [1, 1, 1]])

while True:
      
    # Capture the video frame by frame
    ret, frame = vid.read()
    
    #height,width,depth = frame.shape
    
    #cx = int(width /2)
    #cy = int(height /2)
  
    # Display the resulting frame
    cv2.imshow('frame', frame)
    
    
    # EXERCISE FILTERING
    # # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #converts BGR to gray
    cv2.imshow('gray', gray)
    
    # # Apply sobel filter in x and y directions
    # grad_x = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5) #x gradient
    # grad_y = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5) #y gradient
    
    # # If you want to use your own kernel
    grad_x = cv2.filter2D(gray,-1,kernel_x) #destination depth = -1 so that it is same as input
    
    # # # scale the gradient to -1 to 1 range
    grad_x_normalized = grad_x / np.absolute(grad_x).max()
    grad_x_normalized = np.uint8(128+127*grad_x_normalized)
    
    cv2.imshow('Gradient X',grad_x_normalized)
    # # cv2.imshow('Gradient Y',grad_y)
     
    # frame_thresholded = np.zeros_like(grad_x_normalized,dtype=np.uint8())
    # frame_thresholded[grad_x_normalized>210] = 255
    
    # cv2.imshow('frame thresholded',frame_thresholded)
    
    # EXERCISE COLOR CHANNELS
    
    #blue = frame[:,:,0]
    #green = frame[:,:,1]
    #red = frame[:,:,2]
    
    #cv2.imshow('blue',blue)
    #cv2.imshow('green',green)
    #cv2.imshow('red',red)
    
    # EXERCISE CROPPING
    #frame_crop = frame[cy-50:cy+50,cx-100:cx+100,:]
    #cv2.imshow('cropped frame',frame_crop)
    
    
    # EXERCISE RESIZING
    #frame_crop2 = cv2.resize(frame_crop,(250,250))
    #cv2.imshow('cropped frame resized',frame_crop2)
    
    # Wait key for 1ms, if it is 'q' quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the video capture object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
for i in range(5):
    cv2.waitKey(1) 
