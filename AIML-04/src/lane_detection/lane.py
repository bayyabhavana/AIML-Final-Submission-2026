
import cv2
import numpy as np
from pathlib import Path

# Project directory
ROOT = Path(__file__).resolve().parents[2]

# Input and output paths
IMAGE_PATH = ROOT / "datasets" / "road.jpg"
OUTPUT_DIR = ROOT / "outputs" / "lane_detection"


def region_of_interest(image):
    height, width = image.shape[:2]

    mask = np.zeros_like(image)

    polygon = np.array([[
        (int(width * 0.08), height),
        (int(width * 0.43), int(height * 0.58)),
        (int(width * 0.57), int(height * 0.58)),
        (int(width * 0.95), height)
    ]], dtype=np.int32)

    cv2.fillPoly(mask, polygon, 255)

    return cv2.bitwise_and(image, mask)


def detect_lanes():

    # Check image
    if not IMAGE_PATH.exists():
        print("Error: Road image not found!")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        print("Error: Unable to read road image!")
        return

    print("Road image loaded successfully!")

    # Step 1: Grayscale conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 3: Canny edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Step 4: Region of interest
    roi = region_of_interest(edges)

    # Step 5: Hough line detection
    lines = cv2.HoughLinesP(
        roi,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=30,
        maxLineGap=100
    )

    # Draw detected lines
    output = image.copy()
    count = 0

    if lines is not None:

        for line in lines:

            # Handle OpenCV output formats
            coordinates = np.asarray(line).reshape(-1)

            if coordinates.size < 4:
                continue

            x1, y1, x2, y2 = map(
                int, coordinates[:4]
            )

            cv2.line(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                5
            )

            count += 1

    # Save results
    cv2.imwrite(
        str(OUTPUT_DIR / "lane_result.jpg"),
        output
    )

    cv2.imwrite(
        str(OUTPUT_DIR / "edges.jpg"),
        edges
    )

    cv2.imwrite(
        str(OUTPUT_DIR / "roi.jpg"),
        roi
    )

    print("\nLane detection completed successfully!")
    print("Detected line segments:", count)
    print("Output folder:", OUTPUT_DIR)


if __name__ == "__main__":
    detect_lanes()
