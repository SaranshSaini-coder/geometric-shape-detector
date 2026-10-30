# Geometric Shape Detector & Measurements

A Python application built using OpenCV that detects geometric shapes in images, draws their contours, classifies their types, and calculates essential geometric properties.

## Features
* **Image Preprocessing:** Includes grayscale conversion, Gaussian blur, binary thresholding, and Canny edge detection.
* **Contour Detection:** Finds object boundaries and filters out small contours caused by noise.
* **Polygon Approximation & Classification:** Classifies triangles, squares, rectangles, hexagons, quadrilaterals, and general polygons using the Douglas-Peucker algorithm (`approxPolyDP`).
* **Geometric Measurements:** Computes exact area, perimeter, and bounding box dimensions.

## Requirements
* Python 3.x
* OpenCV (`cv2`)
* NumPy

## Setup & Execution
1. Install dependencies:
   ```bash
   pip install opencv-python numpy
   ```
2. Run the script:
   ```bash
   python shape_detector.py
   ```
