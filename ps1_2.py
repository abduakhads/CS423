# problem2
import cv2
import numpy as np
import time

def process_video():
    cap = cv2.VideoCapture(1)
    # cap = cv2.VideoCapture('input/1.mp4')
    
    if not cap.isOpened():
        raise IOError
    
    edge_enhancement_factor = 1.5
    
    canny_threshold1 = 50
    canny_threshold2 = 150
    
    prev_time = time.time()
    fps = 0
    
    print("Pleqse press 'q' to quit.")
    print(f"enhancement edge factor: {edge_enhancement_factor}")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # (a)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # (b)
        edges = cv2.Canny(gray, canny_threshold1, canny_threshold2)
        
        # (c)
        mask = edges
        
        # (d)
        masked_frame = frame.copy()
        inverted_mask = cv2.bitwise_not(mask)
        darkened = cv2.addWeighted(frame, 0.3, np.zeros_like(frame), 0, 0)
        masked_frame = cv2.bitwise_and(darkened, darkened, mask=inverted_mask)
        masked_frame = cv2.bitwise_or(masked_frame, cv2.bitwise_and(frame, frame, mask=mask))
        
        # (e)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edge_enhanced = cv2.addWeighted(frame, 1, edges_bgr, edge_enhancement_factor, 0)
        
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # (f)
        original_labeled = frame.copy()
        masked_labeled = masked_frame.copy()
        enhanced_labeled = edge_enhanced.copy()
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        text_color = (0, 255, 0)
        
        cv2.putText(original_labeled, "original", (10, 30), font, font_scale, text_color, font_thickness)
        cv2.putText(masked_labeled, "masked", (10, 30), font, font_scale, text_color, font_thickness)
        cv2.putText(enhanced_labeled, "Edge-Enhanced", (10, 30), font, font_scale, text_color, font_thickness)
        
        concatenated = np.hstack((original_labeled, masked_labeled, enhanced_labeled))
        
        # (g)
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(concatenated, fps_text, (10, concatenated.shape[0] - 20), 
                    font, font_scale, (0, 255, 255), font_thickness)
        
        cv2.imshow('compare', concatenated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("stopped")

if __name__ == "__main__":
    process_video()
