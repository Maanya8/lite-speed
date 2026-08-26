"""
test_satellite_pose.py

Run a trained MobileNetV2 + FPN satellite keypoint model on arbitrary images.

For each input image this script:
  1. Resizes it to 240x384, matching the training code.
  2. Applies the same ImageNet normalization.
  3. Loads best_model.pt.
  4. Predicts 11 heatmaps + visibility logits.
  5. Finds the peak (argmax) of every predicted heatmap.
  6. Saves:
       - a heatmap grid (one heatmap per keypoint)
       - an overlay of all heatmaps on the original image
       - an overlay with the predicted peak locations marked
       - a combined figure containing all of the above

Example:
    python test_satellite_pose.py ^
        --checkpoint checkpoints/best_model.pt ^
        --input_dir C:/path/to/test_images ^
        --output_dir test_results

Or for one image:
    python test_satellite_pose.py ^
        --checkpoint checkpoints/best_model.pt ^
        --input_dir C:/path/to/test_image.jpg ^
        --output_dir test_results
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from final_train import MobileNetMultiTaskNet


# ============================================================
# Configuration
# ============================================================

IMAGE_HEIGHT = 240
IMAGE_WIDTH = 384
NUM_KEYPOINTS = 11

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

# ============================================================
# Checkpoint loading
# ============================================================

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


# ============================================================
# Image utilities
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


def resize_for_overlay(original_rgb, target_size):
    """
    Resize the original image to the model/heatmap resolution so
    heatmaps and peak coordinates line up exactly.
    target_size = (H, W)
    """
    h, w = target_size
    image = Image.fromarray(original_rgb)
    image = image.resize(
        (w, h),
        Image.Resampling.BILINEAR
    )

    return np.asarray(image).astype(np.float32) / 255.0


# ============================================================
# Peak extraction
# ============================================================

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


# ============================================================
# Plotting
# ============================================================

def save_visualizations(
    original_rgb,
    heatmaps,
    peaks,
    visibility_probs,
    output_path,
    image_name,
):
    """
    Creates one combined figure containing:

        Top:
            Original image with predicted keypoint peaks.

        Middle:
            All 11 heatmaps.

        Bottom:
            All heatmaps superimposed on the image.

    Also saves separate:
        *_peaks.png
        *_heatmaps.png
        *_overlay.png
        *_combined.png
    """

    K, H, W = heatmaps.shape

    image_resized = resize_for_overlay(
        original_rgb,
        (H, W)
    )

    # --------------------------------------------------------
    # 1. Heatmap grid
    # --------------------------------------------------------

    cols = 4
    rows = int(np.ceil(K / cols))

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(16, 4 * rows)
    )

    axes = np.asarray(axes).reshape(-1)

    for k in range(K):
        ax = axes[k]

        im = ax.imshow(
            heatmaps[k],
            cmap="jet"
        )

        x, y, peak = peaks[k]

        ax.scatter(
            x,
            y,
            s=80,
            marker="x",
            linewidths=2,
            color="white",
        )

        ax.set_title(
            f"KP {k}: peak={peak:.3f}"
        )

        ax.axis("off")

        fig.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04
        )

    for k in range(K, len(axes)):
        axes[k].axis("off")

    fig.suptitle(
        f"Predicted heatmaps - {image_name}",
        fontsize=16
    )

    fig.tight_layout()

    heatmap_path = output_path / f"{image_name}_heatmaps.png"

    fig.savefig(
        heatmap_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    # --------------------------------------------------------
    # 2. Original image + peak locations
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    ax.imshow(image_resized)

    for k, (x, y, peak) in enumerate(peaks):
        visible_prob = visibility_probs[k]

        ax.scatter(
            x,
            y,
            s=100,
            marker="x",
            linewidths=3,
        )

        ax.text(
            x + 5,
            y - 5,
            f"{k} ({peak:.2f})",
            fontsize=10,
            fontweight="bold",
            bbox=dict(
                facecolor="white",
                alpha=0.7,
                edgecolor="none"
            ),
        )

    ax.set_title(
        "Predicted keypoint peaks\n"
        "number = keypoint ID, value = heatmap peak"
    )

    ax.axis("off")

    fig.tight_layout()

    peaks_path = output_path / f"{image_name}_peaks.png"

    fig.savefig(
        peaks_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    # --------------------------------------------------------
    # 3. Combined heatmap overlay
    # --------------------------------------------------------
    #
    # Instead of simply summing all heatmaps, we create a
    # maximum-over-keypoints heatmap. This lets the strongest
    # response at every pixel appear in the overlay.
    # --------------------------------------------------------

    combined_heatmap = np.max(
        heatmaps,
        axis=0
    )

    # Normalize only for visualization.
    hm_min = combined_heatmap.min()
    hm_max = combined_heatmap.max()

    if hm_max > hm_min:
        combined_norm = (
            combined_heatmap - hm_min
        ) / (
            hm_max - hm_min
        )
    else:
        combined_norm = np.zeros_like(
            combined_heatmap
        )

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    ax.imshow(image_resized)

    ax.imshow(
        combined_norm,
        cmap="jet",
        alpha=0.45,
    )

    for k, (x, y, peak) in enumerate(peaks):
        ax.scatter(
            x,
            y,
            s=100,
            marker="x",
            linewidths=3,
        )

        ax.text(
            x + 5,
            y - 5,
            str(k),
            fontsize=11,
            fontweight="bold",
            bbox=dict(
                facecolor="white",
                alpha=0.75,
                edgecolor="none"
            ),
        )

    ax.set_title(
        "All predicted heatmaps superimposed on image"
    )

    ax.axis("off")

    fig.tight_layout()

    overlay_path = output_path / f"{image_name}_overlay.png"

    fig.savefig(
        overlay_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    # --------------------------------------------------------
    # 4. Combined figure
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(18, 14)
    )

    # Large image with peaks.
    ax1 = fig.add_subplot(2, 1, 1)

    ax1.imshow(image_resized)

    for k, (x, y, peak) in enumerate(peaks):
        ax1.scatter(
            x,
            y,
            s=100,
            marker="x",
            linewidths=3,
        )

        ax1.text(
            x + 5,
            y - 5,
            f"{k}: {peak:.2f}",
            fontsize=10,
            fontweight="bold",
            bbox=dict(
                facecolor="white",
                alpha=0.7,
                edgecolor="none"
            ),
        )

    ax1.set_title(
        "Input image + predicted heatmap peaks"
    )

    ax1.axis("off")

    # Overlay.
    ax2 = fig.add_subplot(2, 1, 2)

    ax2.imshow(image_resized)

    ax2.imshow(
        combined_norm,
        cmap="jet",
        alpha=0.45,
    )

    for k, (x, y, peak) in enumerate(peaks):
        ax2.scatter(
            x,
            y,
            s=100,
            marker="x",
            linewidths=3,
        )

        ax2.text(
            x + 5,
            y - 5,
            str(k),
            fontsize=11,
            fontweight="bold",
            bbox=dict(
                facecolor="white",
                alpha=0.75,
                edgecolor="none"
            ),
        )

    ax2.set_title(
        "Heatmaps superimposed on input image"
    )

    ax2.axis("off")

    fig.suptitle(
        f"Satellite keypoint prediction - {image_name}",
        fontsize=18
    )

    fig.tight_layout()

    combined_path = output_path / f"{image_name}_combined.png"

    fig.savefig(
        combined_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    return {
        "heatmaps": heatmap_path,
        "peaks": peaks_path,
        "overlay": overlay_path,
        "combined": combined_path,
    }


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


# ============================================================
# Find input images
# ============================================================

def get_image_paths(input_path):
    input_path = Path(input_path)

    if input_path.is_file():
        return [input_path]

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input path does not exist:\n{input_path}"
        )

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
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


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Test the satellite keypoint model and "
            "visualize predicted heatmaps/peaks."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/best_model.pt",
        help="Path to best_model.pt",
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to one image OR a directory of images",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_results",
        help="Directory where visualizations are saved",
    )

    args = parser.parse_args()

    device = DEVICE

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load model.
    model, num_keypoints, heatmap_size = load_model(
        args.checkpoint,
        device
    )

    image_paths = get_image_paths(
        args.input_dir
    )

    print()
    print(
        f"Found {len(image_paths)} image(s)."
    )
    print(
        f"Saving results to: {output_dir}"
    )
    print()

    for image_number, image_path in enumerate(
        image_paths,
        start=1
    ):
        print(
            f"[{image_number}/{len(image_paths)}] "
            f"{image_path.name}"
        )

        try:
            (
                original_rgb,
                heatmaps,
                peaks,
                visibility_probs,
            ) = predict_image(
                model,
                image_path,
                device,
            )

            image_name = image_path.stem

            paths = save_visualizations(
                original_rgb=original_rgb,
                heatmaps=heatmaps,
                peaks=peaks,
                visibility_probs=visibility_probs,
                output_path=output_dir,
                image_name=image_name,
            )

            keypoints_path = save_keypoint_predictions(
                output_path=output_dir,
                image_name=image_name,
                peaks=peaks,
                visibility_probs=visibility_probs,
                image_size_wh=(original_rgb.shape[1], original_rgb.shape[0]),
            )

            # Print predictions in image coordinates.
            # These coordinates are in the resized 240x384
            # model/heatmap coordinate system.
            print(
                "  Predicted peaks "
                "(x, y, heatmap_peak, visibility_prob):"
            )

            for k, (
                (x, y, peak),
                vis_prob,
            ) in enumerate(
                zip(peaks, visibility_probs)
            ):
                print(
                    f"    kp{k:02d}: "
                    f"x={x:3d}, "
                    f"y={y:3d}, "
                    f"peak={peak:.4f}, "
                    f"visible={vis_prob:.3f}"
                )

            print(
                f"  Saved combined: "
                f"{paths['combined']}"
            )
            print(
                f"  Saved keypoints: "
                f"{keypoints_path}"
            )
            print()

        except Exception as e:
            print(
                f"  ERROR processing {image_path.name}: "
                f"{e}"
            )
            print()

    print("Done.")


if __name__ == "__main__":
    main()