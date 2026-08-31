import json
import math
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from visibility import compute_visibility
from verify import verify, visualize_heatmap



images_dir = Path(r".\speed\speed\images\train") 
labels_path = Path(r".\speed\speed\train.json") 
heatmaps_dir = Path(r".\speed\heatmaps\train") 
heatmaps_dir.mkdir(parents=True, exist_ok=True)
visibility_json_path = heatmaps_dir / "visibility.json"
camera_path = Path(__file__).with_name("camera.json")

keypoints_3d = np.array([
	[-0.37, -0.385, 0.3215],
	[-0.37, 0.385, 0.3215],
	[0.37, 0.385, 0.3215],
	[0.37, -0.385, 0.3215],
	[0.37, 0.304, 0],
	[0.37, -0.264, 0],
    [-0.37, -0.264, 0],
    [-0.37, 0.304, 0],
	[-0.5427, 0.4877, 0.2535],
	[0.5427, 0.4877, 0.2591],
	[0.305, -0.579, 0.2515],
], dtype=np.float32)
#8,9,10 = antenna end
#0-7 corners of cuboid satellite

box_faces = [
    (0, 1, 2),
    (0, 2, 3),
    (6, 7, 4),
    (6, 4, 5),
    (0, 1, 7),
    (0, 7, 6),
    (1, 2, 4),
    (1, 4, 7),
    (2, 3, 5),
    (2, 5, 4),
    (3, 0, 6),
    (3, 6, 5),
]


class HeatmapGenerationError(RuntimeError):
    pass


class LabelsLoadError(HeatmapGenerationError):
    pass


class CameraConfigError(HeatmapGenerationError):
    pass


class ImageLoadError(HeatmapGenerationError):
    pass


class LabelFormatError(HeatmapGenerationError):
    pass


class HeatmapWriteError(HeatmapGenerationError):
    pass


class VisibilityWriteError(HeatmapGenerationError):
    pass


class VisibilityFormatError(HeatmapGenerationError):
    pass


def _load_json_file(path, error_type, context):
    if not path.exists():
        raise error_type(f"Missing {context}: {path}")

    try:
        with path.open("r", encoding="utf-8") as file_handle:
            return json.load(file_handle)
    except json.JSONDecodeError as exc:
        raise error_type(f"Invalid JSON in {context}: {path}") from exc
    except OSError as exc:
        raise error_type(f"Unable to read {context}: {path}") from exc


def load_visibility_map(path):
    if not path.exists():
        return {}

    data = _load_json_file(path, VisibilityFormatError, "visibility map")
    if not isinstance(data, dict):
        raise VisibilityFormatError(f"Visibility map must be a JSON object: {path}")

    return data


def _atomic_write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
    ) as temp_file:
        temp_path = Path(temp_file.name)
        temp_file.write(text)

    try:
        os.replace(temp_path, path)
    except OSError as exc:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise VisibilityWriteError(f"Failed to write file atomically: {path}") from exc


def save_visibility_map(path, visibility_map):
    try:
        serialized = json.dumps(visibility_map, indent=2)
    except TypeError as exc:
        raise VisibilityWriteError(f"Visibility map contains non-serializable data: {path}") from exc

    try:
        _atomic_write_text(path, serialized)
    except OSError as exc:
        raise VisibilityWriteError(f"Failed to save visibility map: {path}") from exc


def save_heatmap(path, heatmap):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=path.parent, prefix=path.name + ".", suffix=".tmp") as temp_file:
        temp_path = Path(temp_file.name)

    try:
        with temp_path.open("wb") as file_handle:
            np.save(file_handle, heatmap)
        os.replace(temp_path, path)
    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise HeatmapWriteError(f"Failed to write heatmap: {path}") from exc


with labels_path.open("r", encoding="utf-8") as file_handle:
    try:
        labels = json.load(file_handle)
    except json.JSONDecodeError as exc:
        raise LabelsLoadError(f"Invalid JSON in labels file: {labels_path}") from exc

with camera_path.open("r", encoding="utf-8") as file_handle:
    try:
        camera_cfg = json.load(file_handle)
    except json.JSONDecodeError as exc:
        raise CameraConfigError(f"Invalid JSON in camera config: {camera_path}") from exc

if not isinstance(labels, list):
    raise LabelsLoadError(f"Expected labels file to contain a JSON list: {labels_path}")

if not isinstance(camera_cfg, dict):
    raise CameraConfigError(f"Expected camera config to contain a JSON object: {camera_path}")

if "cameraMatrix" not in camera_cfg or "distCoeffs" not in camera_cfg:
    raise CameraConfigError(f"Camera config missing required keys: {camera_path}")

camera_matrix = np.array(camera_cfg["cameraMatrix"], dtype=np.float32)
dist_coeffs = np.array(camera_cfg["distCoeffs"], dtype=np.float32)

labels_by_filename = {}
for entry in labels:
    if not isinstance(entry, dict):
        raise LabelsLoadError("Each label entry must be a JSON object")
    filename = entry.get("filename")
    if filename is not None:
        labels_by_filename[filename] = entry

scaling = 0.2


def expected_heatmap_paths(filename):
    source_name = Path(filename).stem
    return [heatmaps_dir / f"{source_name}_point{point_idx:02d}.npy" for point_idx in range(11)]


def load_existing_visibility_entry(visibility_map, filename):
    if filename not in visibility_map:
        return None

    entry = visibility_map[filename]
    if not isinstance(entry, list):
        raise VisibilityFormatError(f"Visibility entry for {filename} must be a list")
    if len(entry) != 11:
        raise VisibilityFormatError(f"Visibility entry for {filename} must contain 11 values")
    if not all(isinstance(value, int) for value in entry):
        raise VisibilityFormatError(f"Visibility entry for {filename} must contain integers")

    return entry


def create_2d_points(q_vbs2tango, r_Vo2To_vbs_true):
    if q_vbs2tango is None or r_Vo2To_vbs_true is None:
        raise LabelFormatError("Quaternion and translation must be present in the label")

    q = np.asarray(q_vbs2tango, dtype=np.float32)
    t = np.asarray(r_Vo2To_vbs_true, dtype=np.float32)

    if q.shape != (4,):
        raise LabelFormatError(f"q_vbs2tango must have shape (4,), got {q.shape}")
    if t.shape != (3,):
        raise LabelFormatError(f"r_Vo2To_vbs_true must have shape (3,), got {t.shape}")

    q_scipy = [q[1], q[2], q[3], q[0]]
    rot_matrix = R.from_quat(q_scipy).as_matrix()
    rvec, _ = cv2.Rodrigues(rot_matrix)
    tvec = t.reshape(3, 1)

    points_2d, _ = cv2.projectPoints(keypoints_3d, rvec, tvec, camera_matrix, dist_coeffs)
    points_2d = points_2d.reshape(-1, 2)
    return scaling * points_2d


def create_heatmap(cx, cy, height, width, k, dx, dy):
    if height <= 0 or width <= 0:
        raise HeatmapGenerationError(f"Invalid heatmap dimensions: height={height}, width={width}")

    sigma = k * abs(math.sqrt(dx**2 + dy**2))
    if sigma <= 0:
        raise HeatmapGenerationError(f"Computed non-positive sigma: {sigma}")

    y = np.arange(height).reshape(height, 1)
    x = np.arange(width).reshape(1, width)
    return np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))


def all_heatmaps(points_2d, image_height, image_width):
    points_2d = np.asarray(points_2d, dtype=np.float32)
    if points_2d.shape != (11, 2):
        raise HeatmapGenerationError(f"points_2d must have shape (11, 2), got {points_2d.shape}")

    heatmaps = np.zeros((11, image_height, image_width), dtype=np.float32)
    dx = abs(np.max(points_2d[:, 0]) - np.min(points_2d[:, 0]))
    dy = abs(np.max(points_2d[:, 1]) - np.min(points_2d[:, 1]))

    for point_index in range(11):
        heatmaps[point_index, :, :] = create_heatmap(
            points_2d[point_index, 0],
            points_2d[point_index, 1],
            height=image_height,
            width=image_width,
            k=0.09,
            dx=dx,
            dy=dy,
        )

    return heatmaps


def compute_visibility_for_label(label):
    try:
        q_vbs2tango = np.array(label["q_vbs2tango"], dtype=np.float32)
        r_Vo2To_vbs_true = np.array(label["r_Vo2To_vbs_true"], dtype=np.float32)
    except KeyError as exc:
        raise LabelFormatError(f"Label is missing required field: {exc.args[0]}") from exc
    except (TypeError, ValueError) as exc:
        raise LabelFormatError("Label contains invalid numeric data") from exc

    if q_vbs2tango.shape != (4,):
        raise LabelFormatError(f"q_vbs2tango must have 4 values, got {q_vbs2tango.shape}")
    if r_Vo2To_vbs_true.shape != (3,):
        raise LabelFormatError(f"r_Vo2To_vbs_true must have 3 values, got {r_Vo2To_vbs_true.shape}")

    q_scipy = [q_vbs2tango[1], q_vbs2tango[2], q_vbs2tango[3], q_vbs2tango[0]]
    rot_matrix = R.from_quat(q_scipy).as_matrix()
    tvec = r_Vo2To_vbs_true.reshape(3, 1)
    visibility = compute_visibility(keypoints_3d, rot_matrix, tvec, box_faces)

    if visibility.shape != (11,):
        raise VisibilityFormatError(f"Visibility computation returned wrong shape: {visibility.shape}")

    return visibility.astype(int).tolist()


def process_image(filename, visibility_map):
    image_path = images_dir / filename
    if not image_path.exists():
        return False

    label = labels_by_filename.get(filename)
    if label is None:
        raise LabelFormatError(f"Missing label for image: {filename}")

    visibility_entry = load_existing_visibility_entry(visibility_map, filename)
    heatmap_paths = expected_heatmap_paths(filename)
    missing_heatmap_paths = [path for path in heatmap_paths if not path.exists()]

    image = cv2.imread(str(image_path))
    if image is None:
        raise ImageLoadError(f"OpenCV failed to read image: {image_path}")

    image = cv2.resize(image, None, fx=scaling, fy=scaling, interpolation=cv2.INTER_AREA)
    if image is None:
        raise ImageLoadError(f"OpenCV failed to resize image: {image_path}")

    points_2d = create_2d_points(label["q_vbs2tango"], label["r_Vo2To_vbs_true"])

    # if missing_heatmap_paths: #comment out this part if you want to overwrite with new heatmaps
    heatmaps = all_heatmaps(points_2d, image.shape[0], image.shape[1])
    for point_index, path in enumerate(heatmap_paths):
        # if path.exists(): #comment out this part if you want to overwrite with new heatmaps
        #     continue
        save_heatmap(path, heatmaps[point_index])

    if visibility_entry is None:
        visibility_map[filename] = compute_visibility_for_label(label)
        save_visibility_map(visibility_json_path, visibility_map)


    #comment out next part if you want to visualize old heatmaps

    if not missing_heatmap_paths and visibility_entry is not None:
        points_2d = create_2d_points(label["q_vbs2tango"], label["r_Vo2To_vbs_true"])
        heatmaps = all_heatmaps(points_2d, image.shape[0], image.shape[1])
        fig, ax = verify(filename, points_2d, scaling) #for visualization
        print(np.max(np.array(heatmaps[0,:,:])))
        if len(heatmaps) > 0:
            visualize_heatmap(image, heatmaps)
        return False

    #for visualizing new heatmaps just comment out next part

    # fig, ax = verify(filename, points_2d, scaling) #for visualization
    # if len(heatmaps) > 0:
    #     visualize_heatmap(image, heatmaps)

    return True


def heatmap_and_visibility():
    visibility_map = load_visibility_map(visibility_json_path)
    processed_count = 0

    for idx in range(77, 78):
        filename = f"img{idx:06d}.jpg"
        processed = process_image(filename, visibility_map)
        if processed:
            processed_count += 1

    print(processed_count)
    return processed_count


if __name__ == "__main__":
    heatmap_and_visibility()



    