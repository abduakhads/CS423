# Problem Set 1 — Report

## Overview

This report summarizes the approaches, methods, and observations for the two scripts in this assignment: `p1.py` and `p2.py`. The emphasis is on image- and video-processing methods (no code included here). The report describes the data flow, processing steps, representative figures, quantitative and qualitative observations, limitations, and suggested extensions.

## Tiny contract (inputs / outputs / success criteria)

- Inputs: image files placed in the repository `input/` directory (for `p1.py`), and a video file or webcam stream for `p2.py` (example: `input/1.mp4`).
- Outputs: visual artifacts and images saved to `output/` (e.g., processed images and a combined visualization `all_results.png`).
- Error modes: missing or unreadable input files produce an early exit and a printed message indicating the missing resource; invalid video device or file results in an error message and termination of the video-processing loop.
- Success criteria: the scripts complete without runtime errors, the expected output images are present in `output/`, and the produced visualizations clearly demonstrate the described processing steps (resizing, blurring, edges, masking, equalization; and live edge enhancement for video).

## Summary of `p1.py` (Image processing pipeline)

### Goal

`p1.py` implements a compact image-processing pipeline to illustrate basic transformations and visual analysis. The pipeline includes resizing, grayscale conversion, Gaussian smoothing, Canny edge detection, binary masking to highlight edges on the color image, and histogram equalization to improve contrast.

### Data flow and steps

1. Load the first image found in `input/`.
2. Resize the image to a fixed scale (75% of the original) and save the result.
3. Convert the color image to grayscale and apply Gaussian blur (σ used to determine kernel size). The blurred image is saved.
4. Run Canny edge detection on the grayscale image to produce an edge map. The binary edge image is saved.
5. Create a 3-channel mask from the edge map and blend it with the original color image so edges remain bright while the background is dimmed. Save the masked image.
6. Perform histogram equalization on the grayscale image to boost contrast and save the result.
7. Produce a single combined figure showing the original, intermediate steps, and histograms; save this visualization (the code saves the figure under `output/all_results.png`).

### Methods and parameter choices

- Resizing: area-preserving interpolation is used for downsampling (good practice for reducing aliasing when shrinking images).
- Grayscale conversion: standard color-to-luminance transform.
- Gaussian blur: kernel size is computed from σ to ensure an odd kernel and a smooth blur; this helps reduce spurious edge detections.
- Edge detection: Canny algorithm with two thresholds (low, high). The pair of thresholds controls sensitivity to edges.
- Masking: the edge map is converted to a 3-channel mask and blended with the original image so edges remain at full intensity while non-edge pixels are dimmed by a background weight.
- Histogram equalization: a global equalization on the grayscale image to redistribute pixel intensity and improve global contrast.

### Suggested figures (referenced in Results)

- Figure 1 — Grid visualization (`output/all_results.png`): original, resized, grayscale, blurred, edges, masked result, equalized image, original histogram, equalized histogram.
- Figure 2 — Close-up comparison: original grayscale histogram vs equalized histogram (highlighting redistribution of intensities).
  - (Recommended) `output/histograms.png` — a dedicated figure showing the original and equalized histograms side-by-side (useful as a zoomed close-up of Figure 1).
- Figure 3 — Edge mask overlay on color image (cropped sample to show edge fidelity and potential artifacts).

### Observations and discussion

- Resizing preserved the overall structure and reduced computation for downstream steps.
- Gaussian blur with moderate σ (e.g., σ=2) reduced high-frequency noise while retaining main contours; however, excessive blur removed fine details that could be meaningful edges.
- Canny detection effectively captures strong contours; threshold choices matter — too low produces noisy response, too high misses faint edges. The pipeline used moderately conservative thresholds to produce clear contours for visualization.
- Masking (bright edges, dim background) is visually effective for emphasizing boundaries in cluttered scenes. However, thin edges and small isolated noise may appear disconnected depending on blur and threshold parameters.
- Histogram equalization visibly widened contrast for images with narrow dynamic range; in images with already good contrast, equalization can introduce unnatural quantization or over-amplify noise.

### Limitations and failure modes

- The pipeline assumes a reasonably sized input image. Extremely small images may lose meaning after resizing; extremely large images can consume significant memory and slow processing.
- Global histogram equalization can be suboptimal for scenes with multiple independent lighting conditions; local methods (e.g., CLAHE) can work better in those cases.
- The pipeline's parameter choices (blur σ, Canny thresholds, mask weights) are fixed in the script; real applications would benefit from adaptively tuning these values per image.

## Summary of `p2.py` (Video / live stream processing)

### Goal

`p2.py` captures a live video stream (or reads from a provided video file) and produces a side-by-side display of: original frame, masked frame (edges emphasized, background dimmed), and an edge-enhanced frame where edge intensities are blended back into the original, creating an enhanced visualization in real time.

### Data flow and steps

1. Open a video source (webcam or `input/1.mp4`).
2. For each frame: convert to grayscale, run Canny edge detection, produce a binary edge mask.
3. Build a masked frame: darken non-edge regions and combine them with full-brightness edge pixels to highlight contours.
4. Edge enhancement: convert the edge mask to a 3-channel image and blend with the original frame to make edges appear stronger.
5. Concatenate the original, masked, and edge-enhanced frames horizontally for display; annotate with FPS and labels.
6. Repeat until the stream ends or the user quits (e.g., pressing 'q').

### Methods and real-time considerations

- The script computes a running FPS estimate and attempts to process frames fast enough to maintain interactive playback.
- To maintain real-time performance, the pipeline uses efficient OpenCV operations (bitwise masking, weighted blending) and avoids costly per-frame allocations where possible.
- Video input fallback: if a webcam is not available, a video file can be used as the input source.

### Observations and discussion

- Real-time masking and edge boosting provide an immediate visual accentuation of scene structure that can be useful for debugging, artistic effects, or as a preprocessing step for tracking.
- Edge enhancement factor controls how pronounced edges appear; excessive enhancement can make the result look noisy or unnatural.
- On slower hardware, frame-rate drops can lead to reduced interactivity; resolution reduction (downsampling frames) can be used as a practical tradeoff.

### Limitations

- The script assumes an available video source and will terminate if a device or file cannot be opened.
- The per-frame processing is CPU-bound if no hardware acceleration is available; this can become a bottleneck for high-resolution video.

## Results (qualitative)

- `p1.py` produces a compact visual summary showing the effect of each processing stage and how histogram equalization changes intensity distribution. The saved `output/all_results.png` is the recommended figure for quick inspection.
- `p2.py` produces live demonstrations of edge masking and enhancement. Extracted frames (if saved) provide snapshots for analysis similar to those used for `p1.py`.

## Recommendations & extensions

- Parameter tuning: add command-line arguments or a small configuration file to let users pick σ, Canny thresholds, mask/background weights, and resizing scale without editing the script.
- Adaptive methods: explore automatic threshold selection for Canny (e.g., median-based thresholds), or adaptive histogram equalization (CLAHE) for improved local contrast.
- Performance: add an optional downsampling stage for `p2.py` when running on slower machines, and consider multi-threaded frame capture and processing for higher throughput.
- Output artifacts: save representative frames from `p2.py` (timestamped) to `output/` for later offline analysis.
- Evaluation: for objective evaluation, add a small dataset with ground-truth edges or quality metrics (SSIM, PSNR for pre/post equalization) to quantify effects.

## Conclusion

The two scripts illustrate foundational image and video processing techniques: smoothing, edge detection, masking, contrast enhancement, and real-time blending. Together they form a small but effective demonstration suite for visual analysis, with clear places for parameterization, adaptive methods, and performance tuning.

## Figures (where to find them)

- `output/all_results.png` — combined grid visualization produced by `p1.py`.
- Saved intermediate images in `output/` (resized, blurred, edges, masked, equalized) — useful for creating zoomed-in figures for reports or presentations.
- `output/histograms.png` — a focused histogram comparison (original vs equalized) produced from the grayscale images; useful for detailed contrast analysis.
- For `p2.py`, consider saving sample frames (e.g., `output/frame_0001.png`) to produce a figure showing the original, masked, and edge-enhanced frames side-by-side.

## Notes for the instructor / reproducibility

- Place one image file in `input/` (supported formats: JPG, PNG, BMP, TIFF) for `p1.py` to process.
- For `p2.py`, either attach a short video file at `input/1.mp4` or switch the capture index to match the available webcam.
- Both scripts print progress to the console and save outputs to `output/`.

---

_Report generated to accompany `p1.py` and `p2.py`. No code is included in this document._
