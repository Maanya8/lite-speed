import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_FILE = "camera.json"

# Folder containing files such as:
#
# img000081_keypoints.json
# img000082_keypoints.json
# img000083_keypoints.json
#
KEYPOINT_FOLDER = "submission_results"

# Folder where pose results will be saved
OUTPUT_FOLDER = "submission_results"


# ============================================================
# 3D SATELLITE KEYPOINTS
# ============================================================
#
# IMPORTANT:
# These MUST correspond to the exact keypoint ordering
# used by your CNN.
#
# keypoint 0 -> POINTS_3D[0]
# keypoint 1 -> POINTS_3D[1]
# ...
# keypoint 10 -> POINTS_3D[10]
#
# Replace these values with your actual SPEED 3D coordinates.
#
# Units here determine the units of the translation output.
# For example, if these are meters, tvec will be in meters.
#

POINTS_3D = np.array([
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
#0-3 lie on 1 plane




# ============================================================
# RANSAC PARAMETERS
# ============================================================

RANSAC_ITERATIONS = 2000 # Maximum allowed reprojection error in pixels
RANSAC_REPROJECTION_ERROR = 5.0
RANSAC_CONFIDENCE = 0.999


# ============================================================
# OPTIONAL CONFIDENCE FILTER
# ============================================================

# False = give all 11 points to RANSAC
# True  = remove points with low heatmap/visibility confidence
#
# I recommend starting with False because RANSAC itself
# is already designed to reject bad correspondences.

USE_CONFIDENCE_FILTER = False
PEAK_THRESHOLD = 0.20
VISIBILITY_THRESHOLD = 0.50



# ============================================================
# LOAD CAMERA
# ============================================================

def load_camera(camera_file):

    with open(camera_file, "r") as f:
        camera_data = json.load(f)

    camera_matrix = np.array(
        camera_data["cameraMatrix"],
        dtype=np.float64
    )

    dist_coeffs = np.array(
        camera_data["distCoeffs"],
        dtype=np.float64
    ).reshape(-1, 1)

    return camera_matrix, dist_coeffs


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_keypoint_file(
    keypoint_file,
    camera_matrix,
    dist_coeffs
):

    # --------------------------------------------------------
    # Load keypoint JSON
    # --------------------------------------------------------

    with open(keypoint_file, "r") as f:
        data = json.load(f)

    image_name = data["image_name"]
    image_size = data["image_size_wh"]
    image_width = image_size[0]
    image_height = image_size[1]
    keypoints = data["keypoints"]


    # --------------------------------------------------------
    # Extract 2D points
    # --------------------------------------------------------

    points_2d = []
    heatmap_peaks = []
    visibility = []

    for kp in keypoints:

        points_2d.append([
            kp["x"],
            kp["y"]
        ])
        heatmap_peaks.append(
            kp["heatmap_peak"]
        )
        visibility.append(
            kp["visibility"]
        )


    points_2d = np.array(
        points_2d,
        dtype=np.float32
    )

    heatmap_peaks = np.array(
        heatmap_peaks,
        dtype=np.float32
    )

    visibility = np.array(
        visibility,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Check number of points
    # --------------------------------------------------------

    if len(points_2d) != len(POINTS_3D):

        raise ValueError(
            f"{image_name}: "
            f"2D point count = {len(points_2d)}, "
            f"3D point count = {len(POINTS_3D)}"
        )


    # --------------------------------------------------------
    # Optional confidence filtering
    # --------------------------------------------------------

    if USE_CONFIDENCE_FILTER:

        valid = (
            (heatmap_peaks >= PEAK_THRESHOLD)
            &
            (visibility >= VISIBILITY_THRESHOLD)
        )

        points_2d_used = points_2d[valid]
        points_3d_used = POINTS_3D[valid]
        original_indices = np.where(valid)[0]

    else:

        points_2d_used = points_2d
        points_3d_used = POINTS_3D
        original_indices = np.arange(
            len(points_2d)
        )


    # --------------------------------------------------------
    # Need at least 4 points for normal PnP
    # --------------------------------------------------------

    if len(points_2d_used) < 4:

        raise ValueError(
            f"{image_name}: "
            f"Not enough valid keypoints for PnP."
        )


    # --------------------------------------------------------
    # PnP + RANSAC
    # --------------------------------------------------------

    success, rvec, tvec, inliers = cv2.solvePnPRansac(

        objectPoints=points_3d_used,
        imagePoints=points_2d_used,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
        iterationsCount=RANSAC_ITERATIONS,
        reprojectionError=RANSAC_REPROJECTION_ERROR,
        confidence=RANSAC_CONFIDENCE,
        flags=cv2.SOLVEPNP_EPNP
    )


    # --------------------------------------------------------
    # Handle failed pose
    # --------------------------------------------------------

    if not success:

        return {
            "image_name": image_name,
            "image_size_wh": image_size,
            "success": False,
            "error": "PnP + RANSAC failed"
        }


    # --------------------------------------------------------
    # Rotation matrix
    # --------------------------------------------------------

    R, _ = cv2.Rodrigues(rvec)


    # --------------------------------------------------------
    # Euler angles
    # --------------------------------------------------------

    euler = ScipyRotation.from_matrix(R).as_euler('xyz', degrees=True) #converts roatation matrix to euler
    roll = float(euler[0])
    pitch = float(euler[1])
    yaw = float(euler[2])


    # --------------------------------------------------------
    # Reproject points
    # --------------------------------------------------------

    projected_points, _ = cv2.projectPoints(points_3d_used,rvec,tvec,camera_matrix,dist_coeffs)
    projected_points = projected_points.reshape( -1,2)


    # --------------------------------------------------------
    # Reprojection errors
    # --------------------------------------------------------

    reprojection_errors = np.linalg.norm(

        points_2d_used - projected_points,
        axis=1
    )


    # --------------------------------------------------------
    # Inlier information
    # --------------------------------------------------------

    if inliers is not None:

        inlier_indices = inliers.ravel()

        inlier_original_indices = (
            original_indices[inlier_indices]
        )

    else:

        inlier_indices = np.array(
            [],
            dtype=np.int32
        )

        inlier_original_indices = np.array(
            [],
            dtype=np.int32
        )


    # --------------------------------------------------------
    # Create per-keypoint results
    # --------------------------------------------------------

    keypoint_results = []

    for i in range(len(points_2d_used)):

        original_idx = int(
            original_indices[i]
        )
        is_inlier = (
            i in set(inlier_indices.tolist())
        )

        keypoint_results.append({

            "keypoint_id": original_idx,

            "predicted_2d": [
                float(points_2d_used[i][0]),
                float(points_2d_used[i][1])
            ],
            "reprojected_2d": [
                float(projected_points[i][0]),
                float(projected_points[i][1])
            ],
            "reprojection_error_px":
                float(reprojection_errors[i]),
            "heatmap_peak":
                float(heatmap_peaks[original_idx]),
            "visibility":
                float(visibility[original_idx]),
            "ransac_inlier":
                bool(is_inlier)
        })


    # --------------------------------------------------------
    # Create final output dictionary
    # --------------------------------------------------------

    result = {

        "image_name": image_name,
        "image_size_wh": [
            image_width,
            image_height
        ],
        "success": True,


        # ----------------------------------------
        # 6-DoF translation
        # ----------------------------------------

        "translation": {

            "x": float(tvec[0, 0]),
            "y": float(tvec[1, 0]),
            "z": float(tvec[2, 0])
        },


        # ----------------------------------------
        # Rotation
        # ----------------------------------------

        "rotation_matrix": R.tolist(),

        "rotation_vector": [
            float(rvec[0, 0]),
            float(rvec[1, 0]),
            float(rvec[2, 0])
        ],


        # ----------------------------------------
        # Euler angles
        # ----------------------------------------

        "euler_angles_degrees": {

            "roll": roll,
            "pitch": pitch,
            "yaw": yaw
        },


        # ----------------------------------------
        # RANSAC information
        # ----------------------------------------

        "ransac": {

            "iterations": RANSAC_ITERATIONS,

            "reprojection_threshold_px":
                RANSAC_REPROJECTION_ERROR,

            "confidence":
                RANSAC_CONFIDENCE,

            "num_input_points":
                int(len(points_2d_used)),

            "num_inliers":
                int(len(inlier_indices)),

            "num_outliers":
                int(
                    len(points_2d_used)
                    -
                    len(inlier_indices)
                ),

            "inlier_keypoints":
                [
                    int(x)
                    for x in inlier_original_indices
                ]
        },


        # ----------------------------------------
        # Overall errors
        # ----------------------------------------

        "reprojection_error": {

            "mean_px":
                float(np.mean(reprojection_errors)),
            "median_px":
                float(np.median(reprojection_errors)),
            "max_px":
                float(np.max(reprojection_errors))
        },


        # ----------------------------------------
        # Per-keypoint information
        # ----------------------------------------

        "keypoints": keypoint_results
    }


    return result


# ============================================================
# MAIN
# ============================================================

def main():

    # Create output directory
    output_folder = Path(OUTPUT_FOLDER)

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )


    # Load camera once
    camera_matrix, dist_coeffs = load_camera(
        CAMERA_FILE
    )


    # Find all keypoint JSON files
    keypoint_folder = Path(
        KEYPOINT_FOLDER
    )

    keypoint_files = sorted(
        keypoint_folder.glob(
            "*_keypoints.json"
        )
    )


    # Process every file
    for keypoint_file in keypoint_files:

        try:

            result = process_keypoint_file(
                keypoint_file,
                camera_matrix,
                dist_coeffs
            )


            output_name = (
                keypoint_file.name
                .replace(
                    "_keypoints.json",
                    "_pose.json"
                )
            )


            output_path = (
                output_folder /
                output_name
            )


            # ------------------------------------------------
            # Write JSON
            # ------------------------------------------------

            with open(
                output_path,
                "w"
            ) as f:

                json.dump(
                    result,
                    f,
                    indent=4
                )


        except Exception as e:

            # Even errors are written to a file.
            # Nothing is printed to console.

            output_name = (
                keypoint_file.name
                .replace(
                    "_keypoints.json",
                    "_pose.json"
                )
            )

            output_path = (
                output_folder /
                output_name
            )


            error_result = {

                "image_name":
                    keypoint_file.stem.replace(
                        "_keypoints",
                        ""
                    ),

                "success": False,

                "error": str(e)
            }


            with open(
                output_path,
                "w"
            ) as f:

                json.dump(
                    error_result,
                    f,
                    indent=4
                )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()