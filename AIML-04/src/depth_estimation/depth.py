
import torch
import cv2
import numpy as np
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parents[2]

IMAGE_PATH = ROOT / "datasets" / "road.jpg"
OUTPUT_DIR = ROOT / "outputs" / "depth_estimation"


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not IMAGE_PATH.exists():
        print("Error: Input image not found!")
        return

    print("Loading MiDaS model...")

    device = torch.device("cpu")

    # Load lightweight MiDaS model
    model = torch.hub.load(
        "intel-isl/MiDaS",
        "MiDaS_small",
        trust_repo=True
    )

    model.to(device)
    model.eval()

    # Load MiDaS image transformation
    midas_transforms = torch.hub.load(
        "intel-isl/MiDaS",
        "transforms",
        trust_repo=True
    )

    transform = midas_transforms.small_transform

    # Read input image
    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        print("Unable to read input image!")
        return

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    print("Estimating relative depth...")

    input_batch = transform(image_rgb).to(device)

    with torch.no_grad():
        prediction = model(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image_rgb.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # Normalize for visualization
    depth_normalized = cv2.normalize(
        depth_map,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Create colored depth visualization
    depth_colored = cv2.applyColorMap(
        depth_normalized,
        cv2.COLORMAP_MAGMA
    )

    # Save results
    cv2.imwrite(
        str(OUTPUT_DIR / "depth_gray.jpg"),
        depth_normalized
    )

    cv2.imwrite(
        str(OUTPUT_DIR / "depth_colored.jpg"),
        depth_colored
    )

    print("\nMiDaS depth estimation completed!")
    print("Device: CPU")
    print("Output:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
