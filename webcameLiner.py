import cv2
import numpy as np
from matplotlib import pyplot as plt

def apply_contrast_stretching(img):
    # Linear technique: Contrast stretching
    min_val = np.min(img)
    max_val = np.max(img)
    stretched = (img - min_val) * (255.0 / (max_val - min_val))
    return stretched.astype(np.uint8)

def apply_gamma_correction(img, gamma=1.5):
    # Nonlinear technique: Gamma correction
    gamma_corrected = np.power(img / 255.0, gamma) * 255.0
    return gamma_corrected.astype(np.uint8)

def compute_histogram(img):
    # Compute histogram for a single channel (grayscale)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def create_histogram_plot(hist, color='b'):
    # Create a histogram plot
    plt.figure(figsize=(4, 2))
    plt.plot(hist, color=color)
    plt.xlim([0, 256])
    plt.axis('off')
    plt.tight_layout()
    
    # Convert plot to image
    plt.gcf().canvas.draw()
    hist_img = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8)
    hist_img = hist_img.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))
    plt.close()
    
    return cv2.cvtColor(hist_img, cv2.COLOR_RGB2BGR)

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    cv2.namedWindow("Webcam Processing", cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Convert to grayscale for simplicity
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply processing techniques
        linear = apply_contrast_stretching(gray)
        nonlinear = apply_gamma_correction(gray)
        
        # Compute histograms
        hist_original = compute_histogram(gray)
        hist_linear = compute_histogram(linear)
        hist_nonlinear = compute_histogram(nonlinear)
        
        # Create histogram plots
        hist_original_img = create_histogram_plot(hist_original, 'b')
        hist_linear_img = create_histogram_plot(hist_linear, 'g')
        hist_nonlinear_img = create_histogram_plot(hist_nonlinear, 'r')
        
        # Resize all images to the same size
        height, width = gray.shape
        size = (width, height)
        
        # Resize histograms to match height of original image
        hist_height = height // 4
        hist_original_img = cv2.resize(hist_original_img, (width, hist_height))
        hist_linear_img = cv2.resize(hist_linear_img, (width, hist_height))
        hist_nonlinear_img = cv2.resize(hist_nonlinear_img, (width, hist_height))
        
        # Create the output frame
        top_row = np.vstack([
            np.hstack([cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 
                       cv2.cvtColor(linear, cv2.COLOR_GRAY2BGR)]),
            np.hstack([hist_original_img, hist_linear_img])
        ])
        
        bottom_row = np.vstack([
            np.hstack([cv2.cvtColor(nonlinear, cv2.COLOR_GRAY2BGR), 
                       np.zeros((height, width, 3), dtype=np.uint8)]),
            np.hstack([hist_nonlinear_img, 
                       np.zeros((hist_height, width, 3), dtype=np.uint8)])
        ])
        
        combined = np.vstack([top_row, bottom_row])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, "Original", (10, 30), font, 0.8, (0, 255, 0), 2)
        cv2.putText(combined, "Linear: Contrast Stretching", (width + 10, 30), font, 0.8, (0, 255, 0), 2)
        cv2.putText(combined, "Nonlinear: Gamma Correction", (10, height + hist_height + 30), font, 0.8, (0, 255, 0), 2)
        
        # Display the combined frame
        cv2.imshow("Webcam Processing", combined)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
