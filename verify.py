import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def visualize_heatmap(image, heatmaps, points_2d=None, alpha=0.45, cmap="jet"):
    """Show one image with all keypoint heatmaps overlaid."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    num_heatmaps = heatmaps.shape[0]
    cols = min(4, num_heatmaps)
    rows = int(np.ceil(num_heatmaps / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for idx, ax in enumerate(axes):
        ax.axis("off")
        if idx >= num_heatmaps:
            continue

        ax.imshow(image)
        ax.imshow(heatmaps[idx], cmap=cmap, alpha=alpha)
        if points_2d is not None:
            ax.scatter(points_2d[idx, 0], points_2d[idx, 1], c="white", s=20)
        ax.set_title(f"Keypoint {idx}")

    plt.tight_layout()
    plt.show()
    return fig

def verify(filename, points_2d, scaling):
    """Plot 2D keypoints on an image.

    Args:
        filename (str): Image filename, e.g. "img000240real.jpg".
        points_2d (array-like): Nx2 pixel coordinates.

    Returns:
        tuple: (fig, ax) matplotlib figure and axis.
    """
    points_2d = np.asarray(points_2d, dtype=np.float32)
    # points_2d = (1/scaling)*points_2d #upsacle by value it was downscaled by

    if points_2d.ndim != 2 or points_2d.shape[1] != 2:
        raise ValueError("points_2d must have shape (N, 2)")

    # Keep the same image-location logic used in visualize.py.
    image_path = os.path.join(
        "./speed/speed/images/train",
        filename,
    )

    img = cv2.imread(image_path)
    img = cv2.resize(img, None, fx=scaling, fy=scaling, interpolation=cv2.INTER_AREA)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img_rgb)

    for i, (x, y) in enumerate(points_2d):
        ax.scatter(x, y, c="lime", s=50, marker="x")
        ax.text(x + 5, y - 5, str(i), color="yellow", fontsize=9)

    ax.set_title(f"2D keypoints on {filename}")
    ax.axis("off")
    plt.show()

    return fig, ax
