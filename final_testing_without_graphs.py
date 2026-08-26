
import json
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from final_train import MobileNetMultiTaskNet

# ============================================================
# Configuration
# ============================================================

IMAGE_HEIGHT = 240
IMAGE_WIDTH = 384
NUM_KEYPOINTS = 11
CHECKPOINT_PATH = "./checkpoints/best_model.pt"

OUTPUT_DIR = "submission_results"
INPUT_DIR = "../speed/speed/images/real_test"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Same normalization used during training.
TRANSFORM = T.Compose([
    T.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

#Loading checkpoint

def load_model(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    # Your training script saves the model under this key.
    state_dict = checkpoint["model_state_dict"]

    # Use the config saved inside the checkpoint when available.
    saved_config = checkpoint.get("config", {})

    num_keypoints = saved_config.get(
        "num_keypoints",
        NUM_KEYPOINTS
    )

    fpn_channels = saved_config.get(
        "fpn_channels",
        128
    )

    heatmap_height = saved_config.get(
        "heatmap_height",
        IMAGE_HEIGHT
    )

    heatmap_width = saved_config.get(
        "heatmap_width",
        IMAGE_WIDTH
    )

    # If the config contains None, fall back to the standard size.
    if heatmap_height is None:
        heatmap_height = IMAGE_HEIGHT

    if heatmap_width is None:
        heatmap_width = IMAGE_WIDTH

    model = MobileNetMultiTaskNet(
        num_keypoints=num_keypoints,
        pretrained=False,
        fpn_channels=fpn_channels,
        target_heatmap_size=(
            heatmap_height,
            heatmap_width,
        ),
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Number of keypoints: {num_keypoints}")
    print(
        f"Heatmap size: "
        f"{heatmap_height} x {heatmap_width}"
    )

    if "epoch" in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch'] + 1}")

    if "val_metrics" in checkpoint:
        metrics = checkpoint["val_metrics"]

        if "mean_pixel_error" in metrics:
            print(
                f"Validation mean pixel error: "
                f"{metrics['mean_pixel_error']:.3f}px"
            )

        if "mean_pred_heatmap_peak" in metrics:
            print(
                f"Validation mean heatmap peak: "
                f"{metrics['mean_pred_heatmap_peak']:.4f}"
            )

    return model, num_keypoints, (heatmap_height, heatmap_width)


def get_heatmap_peaks(heatmaps):
    """
    heatmaps:
        numpy array of shape (K, H, W)

    Returns:
        peaks:
            list of (x, y, value)
            one entry for every keypoint.
    """

    peaks = []

    for k in range(heatmaps.shape[0]):
        heatmap = heatmaps[k]
        flat_index = np.argmax(heatmap)
        y, x = np.unravel_index(
            flat_index,
            heatmap.shape
        )

        peak_value = float(heatmap[y, x])
        peaks.append(
            (int(x), int(y), peak_value)
        )

    return peaks

def get_image_paths(input_path):
    input_path = Path(input_path)

    if input_path.is_file():
        return [input_path]

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input path does not exist:\n{input_path}"
        )

    extensions = {
        ".jpg"
    }

    image_paths = sorted(
        p for p in input_path.iterdir()
        if p.is_file()
        and p.suffix.lower() in extensions
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No image files found in:\n{input_path}"
        )

    return image_paths

def save_keypoint_predictions(
    output_path,
    image_name,
    peaks,
    visibility_probs,
    image_size_wh,
):
    """Save final keypoint coordinates and visibility values to JSON."""

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    keypoints = []

    for (x, y, peak_value), vis_prob in zip(peaks, visibility_probs):
        keypoints.append({
            "x": float(x),
            "y": float(y),
            "heatmap_peak": float(peak_value),
            "visibility": float(vis_prob),
        })

    payload = {
        "image_name": image_name,
        "image_size_wh": [int(image_size_wh[0]), int(image_size_wh[1])],
        "keypoints": keypoints,
    }

    json_path = output_path / f"{image_name}_keypoints.json"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return json_path


# ============================================================
# Run inference on one image
# ============================================================
def load_image(image_path):
    """
    Returns:
        original_rgb: original image as numpy array, H x W x 3
        model_input: normalized tensor, 1 x 3 x 240 x 384
    """
    image = Image.open(image_path).convert("RGB")
    original_rgb = np.asarray(image).copy()
    model_input = TRANSFORM(image).unsqueeze(0)
    return original_rgb, model_input

@torch.no_grad()
def predict_image(
    model,
    image_path,
    device,
):
    original_rgb, model_input = load_image(
        image_path
    )

    model_input = model_input.to(device)

    pred_heatmaps, pred_vis_logits = model(
        model_input
    )

    # Convert to numpy.
    pred_heatmaps = (
        pred_heatmaps[0]
        .cpu()
        .numpy()
    )

    visibility_probs = (
        torch.sigmoid(pred_vis_logits[0])
        .cpu()
        .numpy()
    )

    # Peak of each heatmap.
    peaks = get_heatmap_peaks(
        pred_heatmaps
    )

    return (
        original_rgb,
        pred_heatmaps,
        peaks,
        visibility_probs,
    )



def main():
    device = DEVICE

    model, num_keypoints, heatmap_size = load_model(
        CHECKPOINT_PATH,
        device
    )

    image_paths = get_image_paths(INPUT_DIR)

    for image_number, image_path in enumerate(
            image_paths,
            start=1
        ):

        print(
            f"[{image_number}/{len(image_paths)}] "
            f"{image_path.name}"
        )

        try:
            (original_rgb, heatmaps,peaks,visibility_probs,) = predict_image(model,image_path,device,)

            image_name = image_path.stem

            keypoints_path = save_keypoint_predictions(
                        output_path=OUTPUT_DIR,
                        image_name=image_name,
                        peaks=peaks,
                        visibility_probs=visibility_probs,
                        image_size_wh=(original_rgb.shape[1], original_rgb.shape[0]),
                    )

        except Exception as e:
            print(
                f"  ERROR processing {image_path.name}: "
                f"{e}"
            )
            print()

    print("Done")


if __name__ == "__main__":
    main()