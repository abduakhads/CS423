import cv2
import numpy as np

from ps2_1 import hough_circles_peaks, draw_circles

def auto_canny(image, sigma=0.33):
    v = np.median(image)
    # automatic Canny thresholds
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    
    return cv2.Canny(image, lower, upper)

def hough_circles_acc_fast(img_edges, min_r, max_r, theta_step=4):
    # we use same logic as in hough_circles_acc but optimize the theta loop

    rows, cols = img_edges.shape
    r_range = max_r - min_r
    H = np.zeros((rows, cols, r_range), dtype=np.uint64)
    
    y_idxs, x_idxs = np.nonzero(img_edges) 
    
    # OPTIMIZATION: Reduce theta resolution (step by 4 or 6 degrees)
    thetas = np.deg2rad(np.arange(0, 360, theta_step))
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)
    
    for r_idx, r in enumerate(range(min_r, max_r)):
        dx = np.round(r * cos_thetas).astype(int)
        dy = np.round(r * sin_thetas).astype(int)
        
        for theta_idx in range(len(thetas)):
            # Calculate potential centers for ALL edge pixels
            a = x_idxs - dx[theta_idx]
            b = y_idxs - dy[theta_idx]
            
            valid_mask = (a >= 0) & (a < cols) & (b >= 0) & (b < rows)
            
            valid_a = a[valid_mask]
            valid_b = b[valid_mask]
            
            H[valid_b, valid_a, r_idx] += 1
            
    return H

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    print("Press 'q' to quit.")

    # -- PARAMETERS ---
    # We downscale the image to this width to make Hough run fast enough for video
    process_width = 320 
    
    # These radii are relative to the small (320px) image
    min_r_small = 15     
    max_r_small = 50     
    
    # Threshold for voting
    vote_threshold = 30
    

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resizing for Speed
        height, width = frame.shape[:2]
        scale = process_width / width
        small_frame = cv2.resize(frame, (process_width, int(height * scale)))
        
        # Preprocessing (Blur + Auto-Canny)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 2) 
        edges = auto_canny(blur)

        # Hough Transform
        acc = hough_circles_acc_fast(edges, min_r_small, max_r_small, theta_step=6)
        circles = hough_circles_peaks(acc, min_r_small, t=vote_threshold, s=3)

        # Scaling results back up to original frame size
        final_circles = []
        for (x, y, r) in circles:
            orig_x = int(x / scale)
            orig_y = int(y / scale)
            orig_r = int(r / scale)
            final_circles.append((orig_x, orig_y, orig_r))
            
            cv2.putText(frame, f"r:{orig_r}", (orig_x, orig_y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw using YOUR imported function
        frame_with_circles = draw_circles(frame, final_circles)

        # Show results
        cv2.imshow('Live Hough (Using ps2_1 functions)', frame_with_circles)
        cv2.imshow('Edges (Debug)', edges)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()