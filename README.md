# Satellite Keypoint Pose Estimation

This project trains a deep learning model to estimate keypoint locations on a satellite from images. The pipeline uses a shared image encoder with a multi-task head to predict:

- heatmaps for each keypoint
- visibility logits for each keypoint

The model is trained on precomputed Gaussian heatmaps and visibility labels generated from the satellite’s 3D geometry.

## Overview

The repository contains:

- data preparation and label generation scripts
- training scripts for single-task and multi-task models
- validation / verification utilities
- inference and testing scripts to visualize predictions
- saved checkpoint outputs under `checkpoints/`

## Architecture overview

The core model follows a shared-backbone, multi-task design:

- a pretrained MobileNetV2 backbone extracts image features
- a lightweight FPN-style decoder combines low- and high-resolution feature maps
- one branch predicts a Gaussian heatmap for each keypoint
- a second branch predicts binary visibility for each keypoint
- the final loss combines heatmap regression with visibility supervision

This setup is useful for satellite pose estimation because the network learns both spatial localization and whether a keypoint is actually visible, which is particularly important under occlusion or partial out-of-frame conditions.

## Project structure

- `heatmap_gen.py` — generates per-keypoint heatmaps from satellite pose labels
- `visibility.py` — computes keypoint visibility from 3D geometry and self-occlusion checks
- `verify.py` — visual verification and plotting utilities
- `train_multitask.py` — main training pipeline for heatmap + visibility prediction
- `train_singletask.py` — alternative single-task training pipeline
- `final_testing.py` — runs inference on images and saves prediction outputs
- `final_testing_without_graphs.py` — inference variant without graph plotting
- `visualize.py` — visualization helpers
- `camera.json` — camera intrinsics used during data generation
- `requirements.txt` — Python dependencies
- `checkpoints/` — trained model weights
- `test_results/` and `submission_results/` — generated outputs

## Dependencies

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The project requires:

- Python 3.10+
- PyTorch
- torchvision
- TensorFlow
- OpenCV
- NumPy
- SciPy
- Matplotlib
- Pillow

## Data layout

The training scripts expect a dataset organized roughly like this:

```text
project_root/
├── camera.json
├── checkpoints/
├── heatmap_dir/
│   ├── img000001_point00.npy
│   ├── img000001_point01.npy
│   ├── ...
│   ├── img000002_point00.npy
│   ├── ...
│   └── visibility.json
├── images/
│   ├── img000001.jpg
│   ├── img000002.jpg
│   ├── ...
└── ...
```

Important conventions:

- images are loaded from `image_dir`
- each keypoint has a matching heatmap file named like `img000001_point00.npy`
- `visibility.json` stores per-image visibility values keyed by filename, for example:

```json
{
  "img000004.jpg": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
}
```

Visibility values are:

- `2` = visible
- `1` = occluded or out of bounds

## Generating heatmaps and visibility

Before training, the project expects precomputed 2D heatmaps and visibility labels. The generation flow is:

```powershell
python heatmap_gen.py
```

This script reads the label metadata and camera configuration, projects the 3D satellite keypoints into 2D, creates Gaussian heatmaps, and writes the corresponding visibility JSON.

## Training

The main training script is:

```powershell
python train_multitask.py
```

This trains a MobileNetV2-based model with a feature pyramid decoder to jointly predict:

- keypoint heatmaps
- visibility scores

You can adjust data paths and hyperparameters directly in the script or via CLI arguments. Checkpoints are saved under `checkpoints/`.

A single-task alternative is also available:

```powershell
python train_singletask.py
```

## Inference and evaluation

Run inference using a trained checkpoint:

```powershell
python final_testing.py --checkpoint checkpoints/best_model.pt --input_dir path\to\images --output_dir test_results
```

The script loads a saved model, predicts keypoint peaks, overlays them on the input image, and saves visualization outputs into the specified directory.

A no-plot variant is also available:

```powershell
python final_testing_without_graphs.py
```

## Checkpoints

The repository includes trained checkpoints under `checkpoints/`:

- `best_model.pt`
- `last_checkpoint.pt`

These are the model files typically used for inference and validation.

## Notes

- Image sizes are configured to match the training data, with the project using a typical resolution of `240 x 384` for the model input.
- The visibility logic uses geometry-based occlusion checks to distinguish visible keypoints from self-occluded ones.
- The `verify.py` and visualization tools are useful for debugging localization quality and checking model output against ground truth.

## Typical workflow

1. Prepare images and labels
2. Generate heatmaps and visibility labels
3. Train the model with `train_multitask.py`
4. Evaluate saved checkpoints with `final_testing.py`
5. Inspect outputs in `test_results/` or `submission_results/`

## License

This repository does not currently include a project-specific license file. Please confirm the intended licensing before distributing or reusing the code outside the local project context.
