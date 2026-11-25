
import cv2
import torch
import numpy as np

from pathlib import Path
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


# Project paths
ROOT = Path(__file__).resolve().parents[2]

IMAGE_PATH = ROOT / "datasets" / "road.jpg"

OPENCV_RESULT = ROOT / "outputs" / "lane_detection" / "lane_result.jpg"

OUTPUT_DIR = ROOT / "outputs" / "lane_segmentation"


# Pretrained Cityscapes segmentation model
MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Starting lane segmentation module...")

    # Check input image
    if not IMAGE_PATH.exists():
        print("Error: Road image not found!")
        return

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        print("Unable to read road image!")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("Road image loaded successfully!")

    # Load pretrained model
    print("Loading SegFormer model...")

    device = torch.device("cpu")

    processor = SegformerImageProcessor.from_pretrained(
        MODEL_NAME
    )

    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_NAME
    )

    model.to(device)
    model.eval()

    print("SegFormer model loaded successfully!")

    # Prepare image
    inputs = processor(
        images=image_rgb,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Perform segmentation
    print("Performing semantic segmentation...")

    with torch.no_grad():

        outputs = model(**inputs)

        logits = torch.nn.functional.interpolate(
            outputs.logits,
            size=image_rgb.shape[:2],
            mode="bilinear",
            align_corners=False
        )

        prediction = logits.argmax(dim=1)[0]

    prediction = prediction.cpu().numpy()

    # Find road class
    labels = model.config.id2label

    road_id = None

    for class_id, class_name in labels.items():

        if class_name.lower() == "road":
            road_id = int(class_id)
            break

    if road_id is None:
        print("Road class not found in model labels!")
        print("Available classes:", labels)
        return

    # Create road segmentation mask
    road_mask = (prediction == road_id).astype(np.uint8) * 255

    # Save segmentation mask
    mask_path = OUTPUT_DIR / "road_mask.png"

    cv2.imwrite(
        str(mask_path),
        road_mask
    )

    # Create colored overlay
    overlay = image.copy()

    overlay[road_mask == 255] = (0, 255, 0)

    overlay = cv2.addWeighted(
        image,
        0.6,
        overlay,
        0.4,
        0
    )

    overlay_path = OUTPUT_DIR / "segmentation_overlay.jpg"

    cv2.imwrite(
        str(overlay_path),
        overlay
    )

    # Compare OpenCV lane detection and segmentation
    if OPENCV_RESULT.exists():

        opencv_image = cv2.imread(str(OPENCV_RESULT))

        if opencv_image is not None:

            opencv_image = cv2.resize(
                opencv_image,
                (image.shape[1], image.shape[0])
            )

            comparison = np.hstack(
                (opencv_image, overlay)
            )

            cv2.imwrite(
                str(OUTPUT_DIR / "comparison.jpg"),
                comparison
            )

            print("OpenCV and SegFormer comparison saved!")

    else:
        print("OpenCV lane result not found.")
        print("Run lane.py first to generate the comparison.")

    print("\nSegmentation completed successfully!")

    print("Model: SegFormer-B0")
    print("Dataset: Cityscapes")
    print("Device: CPU")
    print("Detected class: Road")

    print("Results saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
