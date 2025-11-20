#problem 1

import cv2
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pathlib import Path


def load_image(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError
    return img


def resize_image(img):
    # (a)
    scale=0.75
    height, width = img.shape[:2]
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    print(f"original {width}x{height}")
    print(f"resized {new_width}x{new_height}")
    
    return resized


def grayscale_and_blur(img, sigma=2):
    # (b)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)
    print(f"kernel size={kernel_size}x{kernel_size}")
    
    return gray, blurred


def detect_edges(gray_img, low_threshold=50, high_threshold=150):
    # (c)
    edges = cv2.Canny(gray_img, low_threshold, high_threshold)
    print(f"thresholds: {low_threshold}, {high_threshold}")
    
    return edges


def apply_edge_mask(original_img, edges, edge_color_weight=1.0, background_weight=0.3):
    # (d)
    binary_mask = edges.copy()    
    mask_3channel = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    
    mask_normalized = mask_3channel.astype(float) / 255.0    
    inverse_mask = 1.0 - mask_normalized
    
    result = (original_img.astype(float) * mask_normalized * edge_color_weight + 
              original_img.astype(float) * inverse_mask * background_weight)
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def histogram_equalization(gray_img):
    # (e)
    equalized = cv2.equalizeHist(gray_img)

    return equalized


def display_results(original, resized, gray, blurred, edges, masked, equalized, output_dir):
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    masked_rgb = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Problem 1: Basic Image Operations', fontsize=16, fontweight='bold')

    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    axes[0, 1].imshow(resized_rgb)
    axes[0, 1].set_title('(a) Resized (75%)')
    axes[0, 1].axis('off')
    axes[0, 2].imshow(gray, cmap='gray')
    axes[0, 2].set_title('Grayscale')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(blurred, cmap='gray')
    axes[1, 0].set_title('(b) Gaussian Blur (σ=2)')
    axes[1, 0].axis('off')
    axes[1, 1].imshow(edges, cmap='gray')
    axes[1, 1].set_title('(c) Canny Edge Detection')
    axes[1, 1].axis('off')
    axes[1, 2].imshow(masked_rgb)
    axes[1, 2].set_title('(d) Edge Mask Applied')
    axes[1, 2].axis('off')
    
    axes[2, 0].imshow(equalized, cmap='gray')
    axes[2, 0].set_title('(e) Histogram Equalized')
    axes[2, 0].axis('off')
    # print(gray, "\n\n", gray.ravel())
    axes[2, 1].hist(gray.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7)
    axes[2, 1].set_title('Original Histogram')
    axes[2, 1].set_xlabel('Pixel Intensity')
    axes[2, 1].set_ylabel('Frequency')
    axes[2, 2].hist(equalized.ravel(), bins=256, range=[0, 256], color='green', alpha=0.7)
    axes[2, 2].set_title('Equalized Histogram')
    axes[2, 2].set_xlabel('Pixel Intensity')
    axes[2, 2].set_ylabel('Frequency')

    plt.tight_layout()
    figure_path = output_dir / "all_results.png"
    plt.savefig(str(figure_path), dpi=150, bbox_inches='tight')


    fig_hist, axes_hist = plt.subplots(1, 2, figsize=(12, 6))
    fig_hist.suptitle('Histograms', fontsize=16, fontweight='bold')

    axes_hist[0].hist(gray.ravel(), bins=256, range=[0, 256], color='orange', alpha=0.7)
    axes_hist[0].set_title('Original Histogram')
    axes_hist[0].set_xlabel('Pixel Intensity')
    axes_hist[0].set_ylabel('Frequency')
    axes_hist[1].hist(equalized.ravel(), bins=256, range=[0, 256], color='green', alpha=0.7)
    axes_hist[1].set_title('Equalized Histogram')
    axes_hist[1].set_xlabel('Pixel Intensity')
    axes_hist[1].set_ylabel('Frequency')
    plt.savefig(str(output_dir / "histograms.png"), dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved: {figure_path}")
    
    # try:
    #     plt.show()
    # except:
    #     plt.close()


def main():
    input_dir = Path(__file__).parent / "input"
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_path = None
    
    for ext in image_extensions:
        images = list(input_dir.glob(ext))
        if images:
            image_path = images[0]
            break
    
    if image_path is None:
        raise FileNotFoundError
    
    print(f"Processing image: {image_path.name}")
    print("=" * 60)
    
    original = load_image(image_path)
    
    # (a) resize image to 75%
    print("\n(a) Resizing image...")
    resized = resize_image(original, scale=0.75)
    resized_path = output_dir / f"resized_{image_path.stem}.png"
    cv2.imwrite(str(resized_path), resized)
    print(f"Saved: {resized_path}")

    # (b) grayscale and Gaussian blur
    print("\n(b) Converting to grayscale and applying Gaussian blur...")
    gray, blurred = grayscale_and_blur(original, sigma=2)
    blurred_path = output_dir / f"blurred_{image_path.stem}.png"
    cv2.imwrite(str(blurred_path), blurred)
    print(f"Saved: {blurred_path}")

    # (c) edge detection
    print("\n(c) Applying Canny edge detection...")
    edges = detect_edges(gray)
    edges_path = output_dir / f"edges_{image_path.stem}.png"
    cv2.imwrite(str(edges_path), edges)
    print(f"Saved: {edges_path}")
    
    # (d) apply edge mask
    print("\n(d) Creating and applying edge mask...")
    masked = apply_edge_mask(original, edges)
    masked_path = output_dir / f"masked_{image_path.stem}.png"
    cv2.imwrite(str(masked_path), masked)
    print(f"Saved: {masked_path}")

    # (e) histogram equalization
    print("\n(e) Performing histogram equalization...")
    equalized = histogram_equalization(gray)
    equalized_path = output_dir / f"equalized_{image_path.stem}.png"
    cv2.imwrite(str(equalized_path), equalized)
    print(f"Saved: {equalized_path}")
    
    print("\n" + "=" * 60)
    print("All operations completed successfully!")
    print(f"Output files saved in: {output_dir}")
    print("\nDisplaying results...")

    cv2.namedWindow("Gaussian Blurred Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Gaussian Blurred Image", blurred)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.namedWindow("Edge Detected Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Edge Detected Image", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.namedWindow("Masked Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Masked Image", masked)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.namedWindow("Equalized Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Equalized Image", equalized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    display_results(original, resized, gray, blurred, edges, masked, equalized, output_dir)


if __name__ == "__main__":
    main()
