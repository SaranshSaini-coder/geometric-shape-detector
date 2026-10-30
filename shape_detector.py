import cv2
import numpy as np
import os

def generate_sample_image(filename="shapes.png"):
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255

    # Drawing shapes for testing
    pts_tri = np.array([[150, 80], [80, 220], [220, 220]], np.int32)
    cv2.fillPoly(img, [pts_tri], (200, 100, 50))

    cv2.rectangle(img, (320, 80), (460, 220), (50, 180, 50), -1)
    cv2.rectangle(img, (550, 90), (740, 210), (50, 100, 220), -1)

    pts_hex = np.array([[150, 340], [210, 375], [210, 445], [150, 480], [90, 445], [90, 375]], np.int32)
    cv2.fillPoly(img, [pts_hex], (180, 50, 180))

    pts_quad = np.array([[340, 350], [450, 350], [490, 470], [300, 470]], np.int32)
    cv2.fillPoly(img, [pts_quad], (0, 165, 255))

    pts_poly = np.array([[610, 340], [670, 340], [710, 380], [710, 440], 
                         [670, 480], [610, 480], [570, 440], [570, 380]], np.int32)
    cv2.fillPoly(img, [pts_poly], (120, 120, 0))

    # Noise points
    cv2.circle(img, (50, 550), 3, (0, 0, 0), -1)
    cv2.circle(img, (400, 550), 4, (0, 0, 0), -1)

    cv2.imwrite(filename, img)

def process_shapes(image_path):
    if not os.path.exists(image_path):
        generate_sample_image(image_path)

    image = cv2.imread(image_path)
    output = image.copy()

    # Image Preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(blurred, 50, 150)

    # Contour Detection & Noise Filtering
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print("=" * 60)
    print(f"{'SHAPE TYPE':<15} | {'AREA (px²)':<12} | {'PERIMETER':<12} | {'BOUNDING BOX'}")
    print("=" * 60)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue

        perimeter = cv2.arcLength(cnt, True)
        
        # Polygon Approximation
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        vertices = len(approx)

        x, y, w, h = cv2.boundingRect(approx)

        # Classification
        if vertices == 3:
            shape_name = "Triangle"
        elif vertices == 4:
            rect_area = w * h
            extent = float(area) / rect_area
            if extent > 0.90:
                aspect_ratio = float(w) / h
                if 0.95 <= aspect_ratio <= 1.05:
                    shape_name = "Square"
                else:
                    shape_name = "Rectangle"
            else:
                shape_name = "Quadrilateral"
        elif vertices == 5:
            shape_name = "Pentagon"
        elif vertices == 6:
            shape_name = "Hexagon"
        else:
            shape_name = f"Polygon ({vertices})"

        print(f"{shape_name:<15} | {area:<12.2f} | {perimeter:<12.2f} | (x={x}, y={y}, w={w}, h={h})")

        # Drawing outputs
        cv2.drawContours(output, [approx], -1, (0, 255, 0), 3)
        cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 1)
        
        label = f"{shape_name} (A:{int(area)}, P:{int(perimeter)})"
        cv2.putText(output, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    print("=" * 60)
    cv2.imwrite("detected_output.png", output)

    cv2.imshow("Threshold View", thresh)
    cv2.imshow("Canny Edges", edges)
    cv2.imshow("Detected Shapes", output)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_shapes("shapes.png")