import argparse
import json
import random
from dataclasses import dataclass, fields, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ============================================================
# 1. Config -- every knob you asked for, in one place
# ============================================================
@dataclass
class Config:

    # --- data paths ---
    image_dir: str = "./speed/speed/images/train"
    heatmap_dir: str = "./speed/heatmaps/train"
    visibility_json: str = "./speed/heatmaps/train/visibility.json"
    checkpoint_dir: str = "./checkpoints_single"

    # --- data shape ---
    num_keypoints: int = 11
    image_height: int = 240
    image_width: int = 384
    # heatmaps are used exactly as stored on disk -- this pipeline never resizes
    # or downsamples them further. Set these to match your precomputed .npy files
    # (e.g. 240 x 384, matching your 5x-downscaled SPEED heatmaps). If left as
    # None, the size is auto-detected from the first .npy file found instead.
    heatmap_height: int = 240
    heatmap_width: int = 384

    # --- split ---
    val_split: float = 0.15
    split_seed: int = 42

    # --- model ---
    pretrained_backbone: bool = True
    fpn_channels: int = 128        # channel width used throughout the FPN decoder's lateral/smoothing convs

    # --- training schedule ---
    batch_size: int = 32
    num_epochs: int = 35
    freeze_epochs: int = 20        # epochs to keep MobileNetV2 backbone frozen before fine-tuning
    lr_frozen: float = 1e-3        # LR while backbone is frozen (head-only training)
    lr_finetune: float = 1e-4      # LR after unfreezing (whole network)
    weight_decay: float = 1e-4

    # --- early stopping ---
    early_stopping_patience: int = 8

    # --- resuming ---
    resume: bool = False           # if True, resume from checkpoint_dir/last_checkpoint.pt if it exists

    # --- logging ---
    verbose_per_keypoint: bool = False  # print per-keypoint pixel error each epoch

    # --- visualization ---
    visualize_every_epoch: bool = True   # save a heatmap-over-image overlay figure after every epoch
    num_vis_samples: int = 4             # how many fixed validation images to visualize each epoch
    vis_dir: str = "./checkpoints_single/visualizations"

    # --- misc ---
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args() -> Config:
    cfg = Config()
    parser = argparse.ArgumentParser(description="Train satellite keypoint heatmap")
    for f in fields(cfg):
        default = getattr(cfg, f.name)
        arg_type = type(default) if default is not None else str
        if f.type == "bool" or isinstance(default, bool):
            parser.add_argument(f"--{f.name}", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=default)
        else:
            parser.add_argument(f"--{f.name}", type=arg_type if default is not None else str, default=default)
    args = parser.parse_args()
    for f in fields(cfg):
        setattr(cfg, f.name, getattr(args, f.name))
    return cfg



# ============================================================
# 2. Dataset
# ============================================================
class SatelliteKeypointDataset(Dataset):
    """
    Loads an image, its num_keypoints precomputed heatmaps, and its visibility vector.

    Visibility is still loaded here (it's cheap and used to mask the heatmap
    loss for occluded keypoints), but the model itself is single-task: it only
    predicts heatmaps, no visibility classification head.

    image_ids: list of stems, e.g. "img000004" (no extension)
    """

    def __init__(self, image_ids, image_dir, heatmap_dir, visibility_dict,
                 num_keypoints, image_size, heatmap_size, augment=False):
        self.image_ids = image_ids
        self.image_dir = Path(image_dir)
        self.heatmap_dir = Path(heatmap_dir)
        self.visibility_dict = visibility_dict
        self.num_keypoints = num_keypoints
        self.image_size = image_size      # (H, W)
        self.heatmap_size = heatmap_size  # (H, W)
        self.augment = augment

        # ImageNet normalization, since we use an ImageNet-pretrained MobileNetV2 backbone
        base_transforms = [
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        if self.augment:
            self.transform = T.Compose([
                T.Resize(self.image_size),
                T.ColorJitter(brightness=0.3, contrast=0.3),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = T.Compose(base_transforms)

    def __len__(self):
        return len(self.image_ids)

    def _load_heatmaps(self, image_id):
        heatmaps = np.zeros((self.num_keypoints, *self.heatmap_size), dtype=np.float32)
        for k in range(self.num_keypoints):
            path = self.heatmap_dir / f"{image_id}_point{k:02d}.npy"
            hm = np.load(path).astype(np.float32)
            if hm.shape != self.heatmap_size:
                raise ValueError(
                    f"{path} has shape {hm.shape}, expected {self.heatmap_size}. "
                    f"Check heatmap_height/heatmap_width in Config."
                )
            heatmaps[k] = hm
        return heatmaps

    def _load_visibility(self, image_id):
        # visibility.json is keyed by filename with extension, e.g. "img000004.jpg"
        key = f"{image_id}.jpg"
        raw = self.visibility_dict[key]  # list of 1s and 2s, length num_keypoints
        # 2 (visible) -> 1.0, 1 (occluded/out of bounds) -> 0.0
        vis = np.array([1.0 if v == 2 else 0.0 for v in raw], dtype=np.float32)
        return vis

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        img_path = self.image_dir / f"{image_id}.jpg"
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        heatmaps = torch.from_numpy(self._load_heatmaps(image_id))
        visibility = torch.from_numpy(self._load_visibility(image_id))

        return image, heatmaps, visibility


#training/validation split + shuffle
def build_train_val_ids(visibility_json_path, val_split, seed):
    with open(visibility_json_path, "r") as f:
        visibility_dict = json.load(f)

    # strip ".jpg" to get stems like "img000004"
    all_ids = [Path(k).stem for k in visibility_dict.keys()]
    rng = random.Random(seed)
    rng.shuffle(all_ids)

    n_val = max(1, int(len(all_ids) * val_split))
    val_ids = all_ids[:n_val]
    train_ids = all_ids[n_val:]
    return train_ids, val_ids, visibility_dict

def autodetect_heatmap_size(heatmap_dir, sample_image_id, num_keypoints=11):
    for k in range(num_keypoints):
        path = Path(heatmap_dir) / f"{sample_image_id}_point{k:02d}.npy"
        if path.exists():
            arr = np.load(path)
            return arr.shape  # (H, W)
    raise FileNotFoundError(
        f"Could not find any heatmap files for '{sample_image_id}' in {heatmap_dir} "
        f"to auto-detect heatmap_height/heatmap_width."
    )


# ============================================================
# 3. Model -- single task: heatmaps only, no visibility head
# ============================================================
class MobileNetKeypointNet(nn.Module):
    """
    MobileNetV2 backbone with an FPN/U-Net-style decoder: instead of decoding
    purely from the deepest (stride-32, ~8x12) feature map, this fuses in
    higher-resolution intermediate features via lateral skip connections
    (stride 4, 8, 16, 32), so fine spatial detail that would otherwise be
    discarded by the time the decoder sees it is preserved and used.

    Single-task: outputs only the K heatmaps. There is no visibility
    classification branch.
    """

    # (layer_index_in_mobilenet_v2.features, output_channels, stride) for the
    # four feature maps we tap into. These indices are fixed by torchvision's
    # mobilenet_v2 architecture.
    SKIP_LAYERS = {
        3: (24, 4),     # stride 4  -- highest resolution we use, ~60x96 for a 240x384 input
        6: (32, 8),     # stride 8
        13: (96, 16),   # stride 16
        18: (1280, 32), # stride 32 -- deepest features, ~8x12 for a 240x384 input
    }

    def __init__(self, num_keypoints=11, pretrained=True, fpn_channels=128,
                 target_heatmap_size=(240, 384)):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights="IMAGENET1K_V2" if pretrained else None)
        # keep the backbone as individual layers (not one nn.Sequential) so we
        # can grab intermediate outputs during the forward pass
        self.backbone_layers = nn.ModuleList(list(mobilenet.features.children()))
        self.target_heatmap_size = target_heatmap_size
        self.deepest_idx = max(self.SKIP_LAYERS.keys())

        # lateral 1x1 convs: project each tapped feature map to a common channel width
        self.lateral_convs = nn.ModuleDict({
            str(idx): nn.Conv2d(ch, fpn_channels, kernel_size=1)
            for idx, (ch, _stride) in self.SKIP_LAYERS.items()
        })
        # 3x3 smoothing conv after each top-down upsample+add (standard FPN practice --
        # reduces the aliasing/checkerboard artifacts that plain upsample+add introduces)
        self.smooth_convs = nn.ModuleDict({
            str(idx): nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
            for idx in self.SKIP_LAYERS if idx != self.deepest_idx
        })

        self.heatmap_head = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_channels, num_keypoints, kernel_size=1),
        )

    def forward(self, x):
        skip_feats = {}
        h = x
        for i, layer in enumerate(self.backbone_layers):
            h = layer(h)
            if i in self.SKIP_LAYERS:
                skip_feats[i] = h
            if i == self.deepest_idx:
                break

        sorted_idx = sorted(self.SKIP_LAYERS.keys(), reverse=True)  # [18, 13, 6, 3]
        fused = self.lateral_convs[str(sorted_idx[0])](skip_feats[sorted_idx[0]])
        for idx in sorted_idx[1:]:
            lateral = self.lateral_convs[str(idx)](skip_feats[idx])
            fused_upsampled = nn.functional.interpolate(
                fused, size=lateral.shape[-2:], mode="bilinear", align_corners=False
            )
            fused = self.smooth_convs[str(idx)](fused_upsampled + lateral)
        # `fused` is at stride 4 (~60x96 for a 240x384 input)

        # heatmap head now runs at stride-4 resolution -- upsample happens after,
        # not before, so the head's convs operate on ~16x fewer pixels
        heatmaps = self.heatmap_head(fused)  # (B, K, stride4_H, stride4_W)
        heatmaps = nn.functional.interpolate(
            heatmaps, size=self.target_heatmap_size, mode="bilinear", align_corners=False
        )

        return heatmaps

    # --- freeze / unfreeze helpers (BatchNorm-aware, see explanation below) ---
    def freeze_backbone(self):
        for layer in self.backbone_layers:
            for param in layer.parameters():
                param.requires_grad = False
            layer.eval()  # freezes BatchNorm running stats too

    def unfreeze_backbone(self):
        for layer in self.backbone_layers:
            for param in layer.parameters():
                param.requires_grad = True
            layer.train()

# ============================================================
# 4. Losses
# ============================================================

def weighted_heatmap_loss(pred, target, alpha=10.0):
    weights = 1.0 + alpha * target
    loss = weights * (pred - target) ** 2
    return loss

def masked_heatmap_loss(pred_heatmaps, target_heatmaps, visibility):
    """MSE over heatmap pixels, masked so occluded keypoints contribute zero gradient."""
    vis_mask = visibility.view(visibility.shape[0], visibility.shape[1], 1, 1)  # (B, K, 1, 1)
    loss_per_pixel = weighted_heatmap_loss(pred_heatmaps, target_heatmaps, alpha=10.0)
    masked_loss = loss_per_pixel * vis_mask
    num_visible = vis_mask.sum().clamp(min=1)
    return masked_loss.sum() / (num_visible * pred_heatmaps.shape[2] * pred_heatmaps.shape[3])


# ============================================================
# 5. Validation metrics
#    - per-keypoint pixel error for visible keypoints
#    - heatmap "collapse" sanity checks (mean/per-kp peak, keypoint spread ratio)
# ============================================================
def soft_argmax_batch(heatmaps):
    """
    heatmaps: (B, K, H, W)
    returns: (B, K, 2) -> (x, y) coordinates in heatmap pixel space
    """
    b, k, h, w = heatmaps.shape
    flat = heatmaps.view(b, k, -1)
    prob = torch.softmax(flat, dim=-1).view(b, k, h, w)

    ys = torch.arange(h, device=heatmaps.device, dtype=heatmaps.dtype)
    xs = torch.arange(w, device=heatmaps.device, dtype=heatmaps.dtype)

    y_coord = (prob.sum(dim=3) * ys).sum(dim=2)  # (B, K)
    x_coord = (prob.sum(dim=2) * xs).sum(dim=2)  # (B, K)
    return torch.stack([x_coord, y_coord], dim=-1)  # (B, K, 2)


def keypoint_spread_ratio(pred_coords, true_coords, eps=1e-6):
    """
    EARLY COLLAPSE DETECTOR. pred_coords, true_coords: (B, K, 2).

    Returns, per image, the ratio of (average pairwise distance between the K
    predicted keypoints) to (average pairwise distance between the K true
    keypoints). Near 1.0 = predicted keypoints are about as spread out as the
    true layout. Near 0 = predicted keypoints are collapsing toward a single
    point regardless of where the true keypoints actually are.

    This catches collapse that mean_pixel_error can miss: if the true
    keypoints on an image happen to sit close together, a collapsed
    (bunched-up) prediction can still score a deceptively low pixel error.
    Spread ratio compares differentiation-between-keypoints directly, so it
    doesn't depend on the true layout being spread out to detect it.
    """
    pred_dist = torch.cdist(pred_coords, pred_coords)  # (B, K, K)
    true_dist = torch.cdist(true_coords, true_coords)  # (B, K, K)

    b, k, _ = pred_dist.shape
    off_diag = ~torch.eye(k, dtype=torch.bool, device=pred_dist.device)
    pred_mean = pred_dist[:, off_diag].view(b, -1).mean(dim=1)  # (B,)
    true_mean = true_dist[:, off_diag].view(b, -1).mean(dim=1)  # (B,)

    return pred_mean / (true_mean + eps)  # (B,)

@torch.no_grad()
def evaluate(model, loader, device, num_keypoints):
    model.eval()
    total_loss = 0.0  # FIX: was declared but never accumulated -> best_val_loss was always 0.0

    # accumulators for per-keypoint pixel error
    per_kp_pixel_error_sum = torch.zeros(num_keypoints, device=device)
    per_kp_visible_count = torch.zeros(num_keypoints, device=device)

    # accumulators for the heatmap-collapse sanity check
    total_pred_peak, total_target_peak, n_samples = 0.0, 0.0, 0
    # PER-KEYPOINT peak accumulators -- the aggregate total_pred_peak above averages
    # across ALL keypoints and images, which can hide one or two collapsed keypoints
    # if the rest are healthy. Tracking per-keypoint catches that.
    per_kp_pred_peak_sum = torch.zeros(num_keypoints, device=device)
    per_kp_target_peak_sum = torch.zeros(num_keypoints, device=device)
    per_kp_peak_count = 0

    # accumulator for the keypoint-spread-ratio collapse check
    total_spread_ratio, spread_ratio_min, n_spread = 0.0, float("inf"), 0

    for images, target_hm, target_vis in loader:
        images, target_hm, target_vis = images.to(device), target_hm.to(device), target_vis.to(device)
        pred_hm = model(images)

        loss = masked_heatmap_loss(pred_hm, target_hm, target_vis)
        total_loss += loss.item()  # FIX

        pred_coords = soft_argmax_batch(pred_hm)
        true_coords = soft_argmax_batch(target_hm)
        pixel_dist = torch.norm(pred_coords - true_coords, dim=-1)  # (B, K)
        mask = target_vis  # (B, K), already 0/1 float

        per_kp_pixel_error_sum += (pixel_dist * mask).sum(dim=0)
        per_kp_visible_count += mask.sum(dim=0)

        # heatmap collapse check: average peak (max) activation, pred vs target
        pred_peak_per_kp = pred_hm.amax(dim=(2, 3))    # (B, K)
        target_peak_per_kp = target_hm.amax(dim=(2, 3))  # (B, K)
        total_pred_peak += pred_peak_per_kp.sum().item()
        total_target_peak += target_peak_per_kp.sum().item()
        n_samples += pred_hm.shape[0] * pred_hm.shape[1]

        # PER-KEYPOINT peak collapse check (catches collapse localized to a
        # subset of keypoints that the aggregate average above can hide)
        per_kp_pred_peak_sum += pred_peak_per_kp.sum(dim=0)
        per_kp_target_peak_sum += target_peak_per_kp.sum(dim=0)
        per_kp_peak_count += pred_hm.shape[0]

        # keypoint-spread-ratio collapse check (catches collapse that
        # mean_pixel_error can miss when true keypoints happen to sit close together)
        spread = keypoint_spread_ratio(pred_coords, true_coords)  # (B,)
        total_spread_ratio += spread.sum().item()
        spread_ratio_min = min(spread_ratio_min, spread.min().item())
        n_spread += spread.shape[0]

    n = len(loader)

    per_kp_pixel_error = (per_kp_pixel_error_sum / per_kp_visible_count.clamp(min=1)).cpu().tolist()

    return {
        "val_loss": total_loss / n,  # FIX: now a real, non-zero value

        "mean_pixel_error": sum(per_kp_pixel_error) / len(per_kp_pixel_error),
        "per_kp_pixel_error": per_kp_pixel_error,

        "mean_pred_heatmap_peak": total_pred_peak / n_samples,
        "mean_target_heatmap_peak": total_target_peak / n_samples,

        # EARLY COLLAPSE DETECTION additions
        "per_kp_pred_peak": (per_kp_pred_peak_sum / per_kp_peak_count).cpu().tolist(),
        "per_kp_target_peak": (per_kp_target_peak_sum / per_kp_peak_count).cpu().tolist(),
        "mean_keypoint_spread_ratio": total_spread_ratio / n_spread,
        "min_keypoint_spread_ratio": spread_ratio_min,
    }


# ============================================================
# 6. Visualization -- predicted heatmap overlaid on the original image
# ============================================================
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def _denormalize_image(img_tensor):
    """img_tensor: (3, H, W) normalized. Returns (H, W, 3) uint8 RGB array."""
    img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


@torch.no_grad()
def make_epoch_visualization(model, vis_images, vis_ids, device, epoch, save_path,
                              image_size, alpha=0.45):
    """
    Renders a grid: one column per sample, showing original image with the
    combined (max-across-keypoints) predicted heatmap overlaid in a jet
    colormap, plus a marker at each keypoint's soft-argmax location.

    vis_images: (N, 3, H, W) tensor, already normalized (same fixed samples every epoch)
    vis_ids: list of image_id strings for titling
    """
    model.eval()
    images = vis_images.to(device)
    pred_hm = model(images)  # (N, K, Hh, Wh)
    pred_coords = soft_argmax_batch(pred_hm)  # (N, K, 2), in heatmap pixel space

    n = images.shape[0]
    img_h, img_w = image_size
    hm_h, hm_w = pred_hm.shape[-2], pred_hm.shape[-1]
    scale_x, scale_y = img_w / hm_w, img_h / hm_h

    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for i in range(n):
        img_np = _denormalize_image(images[i])  # (H, W, 3)

        # combined heatmap: max over all keypoints, then normalize to [0, 1]
        combined = pred_hm[i].amax(dim=0).clamp(min=0).cpu().numpy()  # (Hh, Wh)
        cmax = combined.max()
        if cmax > 1e-8:
            combined = combined / cmax

        ax = axes[i]
        ax.imshow(img_np)
        ax.imshow(combined, cmap=cm.jet, alpha=alpha, extent=(0, img_w, img_h, 0))

        coords = pred_coords[i].cpu().numpy()
        xs = coords[:, 0] * scale_x
        ys = coords[:, 1] * scale_y
        ax.scatter(xs, ys, c="white", edgecolors="black", s=25, linewidths=0.8)

        ax.set_title(vis_ids[i], fontsize=9)
        ax.axis("off")

    fig.suptitle(f"Epoch {epoch + 1}: predicted heatmaps over image", fontsize=12)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def get_fixed_visualization_batch(dataset, num_samples, seed=0):
    """Picks a fixed set of sample indices (same images every epoch) so the
    saved figures form a coherent before/after sequence you can flip through."""
    rng = random.Random(seed)
    n = len(dataset)
    idxs = rng.sample(range(n), min(num_samples, n))
    images, ids = [], []
    for idx in idxs:
        img, _hm, _vis = dataset[idx]
        images.append(img)
        ids.append(dataset.image_ids[idx])
    return torch.stack(images, dim=0), ids


# ============================================================
# 7. Resumable-training checkpoint helpers
# ============================================================
def save_resume_checkpoint(path, epoch, model, optimizer, best_val_loss, epochs_without_improvement, cfg):
    """Saves everything needed to resume training exactly where it left off."""
    torch.save({
        "epoch": epoch,  # last COMPLETED epoch index (0-based)
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "epochs_without_improvement": epochs_without_improvement,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "config": asdict(cfg),
    }, path)


def load_resume_checkpoint(path, device):
    # weights_only=False: this checkpoint is self-generated (not third-party) and
    # includes optimizer/RNG state alongside tensors, which PyTorch's default
    # weights_only=True loader (as of PyTorch 2.6+) does not support deserializing.
    return torch.load(path, map_location=device, weights_only=False)


def restore_rng_state(ckpt):
    torch.set_rng_state(ckpt["torch_rng_state"])
    if ckpt.get("cuda_rng_state") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
    np.random.set_state(ckpt["numpy_rng_state"])
    random.setstate(ckpt["python_rng_state"])

# ============================================================
# 8. Training loop
# ============================================================
def train(cfg: Config):
    set_seed(cfg.seed)
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    resume_path = Path(cfg.checkpoint_dir) / "last_checkpoint.pt"

    train_ids, val_ids, visibility_dict = build_train_val_ids(
        cfg.visibility_json, cfg.val_split, cfg.split_seed
    )
    print(f"Train images: {len(train_ids)} | Val images: {len(val_ids)}")

    heatmap_size = (cfg.heatmap_height, cfg.heatmap_width)
    if cfg.heatmap_height is None or cfg.heatmap_width is None:
        heatmap_size = autodetect_heatmap_size(cfg.heatmap_dir, train_ids[0], cfg.num_keypoints)
        print(f"Auto-detected heatmap size: {heatmap_size}")

    train_ds = SatelliteKeypointDataset(
        train_ids, cfg.image_dir, cfg.heatmap_dir, visibility_dict,
        cfg.num_keypoints, (cfg.image_height, cfg.image_width), heatmap_size, augment=True,
    )
    val_ds = SatelliteKeypointDataset(
        val_ids, cfg.image_dir, cfg.heatmap_dir, visibility_dict,
        cfg.num_keypoints, (cfg.image_height, cfg.image_width), heatmap_size, augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                               num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)

    # fixed batch of validation images used for the per-epoch visualization,
    # so every saved figure shows the same images (a coherent convergence sequence)
    if cfg.visualize_every_epoch:
        vis_images, vis_ids = get_fixed_visualization_batch(val_ds, cfg.num_vis_samples, seed=cfg.seed)

    model = MobileNetKeypointNet(
        num_keypoints=cfg.num_keypoints,
        pretrained=cfg.pretrained_backbone,
        fpn_channels=cfg.fpn_channels,
        target_heatmap_size=heatmap_size,
    ).to(cfg.device)

    start_epoch = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    backbone_frozen = True  # tracked explicitly so we don't re-trigger the freeze->unfreeze transition after resuming

    if cfg.resume and resume_path.exists():
        print(f"Resuming from checkpoint: {resume_path}")
        ckpt = load_resume_checkpoint(resume_path, cfg.device)

        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        epochs_without_improvement = ckpt["epochs_without_improvement"]

        # put the model into whatever freeze/unfreeze stage it was in at that epoch,
        # *before* building the optimizer, so the optimizer's parameter group matches
        # what was saved (this is the part that breaks if done in the wrong order)
        backbone_frozen = start_epoch < cfg.freeze_epochs
        if backbone_frozen:
            model.freeze_backbone()
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=cfg.lr_frozen, weight_decay=cfg.weight_decay,
            )
        else:
            model.unfreeze_backbone()
            optimizer = optim.Adam(model.parameters(), lr=cfg.lr_finetune, weight_decay=cfg.weight_decay)

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        restore_rng_state(ckpt)

        print(f"Resumed at epoch {start_epoch} | best_val_loss={best_val_loss:.5f} | "
              f"epochs_without_improvement={epochs_without_improvement} | "
              f"backbone_frozen={backbone_frozen}")
    else:
        if cfg.resume:
            print(f"--resume was set but no checkpoint found at {resume_path}; starting fresh.")
        # --- stage 1: freeze backbone ---
        model.freeze_backbone()
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.lr_frozen, weight_decay=cfg.weight_decay,
        )

    try:
        for epoch in range(start_epoch, cfg.num_epochs):
            # --- stage 2: unfreeze at the configured epoch (only triggers once) ---
            if backbone_frozen and epoch >= cfg.freeze_epochs:
                print(f"[epoch {epoch}] unfreezing backbone, switching to lr_finetune={cfg.lr_finetune}")
                model.unfreeze_backbone()
                optimizer = optim.Adam(model.parameters(), lr=cfg.lr_finetune, weight_decay=cfg.weight_decay)
                backbone_frozen = False

            model.train()
            if backbone_frozen:
                model.freeze_backbone()  # re-assert BatchNorm eval() each epoch (model.train() above flips it back)

            running_loss = 0.0
            for images, target_hm, target_vis in train_loader:
                images = images.to(cfg.device)
                target_hm = target_hm.to(cfg.device)
                target_vis = target_vis.to(cfg.device)

                optimizer.zero_grad()
                pred_hm = model(images)
                loss = masked_heatmap_loss(pred_hm, target_hm, target_vis)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            n_batches = len(train_loader)
            stage = "frozen" if backbone_frozen else "finetune"
            print(f"[{stage}] epoch {epoch+1}/{cfg.num_epochs} | "
                  f"train_loss={running_loss/n_batches:.5f} ")

            val_metrics = evaluate(model, val_loader, cfg.device, cfg.num_keypoints)

            print(f"           val_loss={val_metrics['val_loss']:.5f}")
            print(f"           mean_pixel_error={val_metrics['mean_pixel_error']:.2f}px")
            print(f"           mean_heatmap_peak: pred={val_metrics['mean_pred_heatmap_peak']:.4f} "
                  f"vs target={val_metrics['mean_target_heatmap_peak']:.4f} "
                  f"{'<-- WARNING: predicted peaks are collapsing toward 0, check for vanishing signal' if val_metrics['mean_pred_heatmap_peak'] < 0.1 * val_metrics['mean_target_heatmap_peak'] else ''}")

            # EARLY COLLAPSE DETECTION: keypoint-spread-ratio. Unlike mean_pixel_error,
            # this doesn't depend on true keypoints being spread out to catch collapse --
            # it directly compares how differentiated the predicted keypoints are from
            # each other vs how differentiated the true keypoints are.
            mean_spread = val_metrics["mean_keypoint_spread_ratio"]
            min_spread = val_metrics["min_keypoint_spread_ratio"]
            spread_warning = (
                " <-- WARNING: predicted keypoints are collapsing toward a single point "
                "(spread ratio near 0 means they're not spatially differentiated)"
                if mean_spread < 0.3 else ""
            )
            print(f"           mean_keypoint_spread_ratio={mean_spread:.4f} "
                  f"(min over batch={min_spread:.4f}, 1.0=healthy, near 0=collapsed){spread_warning}")

            # EARLY COLLAPSE DETECTION: per-keypoint peak check. The aggregate
            # mean_heatmap_peak above averages across all keypoints, which can hide
            # a handful of collapsed keypoints if the rest are fine. Flag any
            # individual keypoint whose peak is far below its target.
            collapsed_kps = [
                k for k, (p, t) in enumerate(zip(val_metrics["per_kp_pred_peak"], val_metrics["per_kp_target_peak"]))
                if p < 0.3 * t
            ]
            if collapsed_kps:
                print(f"           <-- WARNING: keypoint(s) {collapsed_kps} have predicted peaks "
                      f"below 30% of their target peak (possible localized collapse)")

            if cfg.verbose_per_keypoint:
                print("           per-keypoint pixel error (px) / peak (pred vs target):")
                for k, err in enumerate(val_metrics["per_kp_pixel_error"]):
                    p_peak = val_metrics["per_kp_pred_peak"][k]
                    t_peak = val_metrics["per_kp_target_peak"][k]
                    print(f"             kp{k:02d}: pixel_error={err:6.2f} | "
                          f"peak: pred={p_peak:.3f} vs target={t_peak:.3f}")

            # --- per-epoch visualization: predicted heatmap over the original image ---
            if cfg.visualize_every_epoch:
                vis_save_path = Path(cfg.vis_dir) / f"epoch_{epoch+1:03d}.png"
                make_epoch_visualization(
                    model, vis_images, vis_ids, cfg.device, epoch, vis_save_path,
                    image_size=(cfg.image_height, cfg.image_width),
                )
                print(f"           -> saved visualization to {vis_save_path}")

            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                epochs_without_improvement = 0
                best_path = Path(cfg.checkpoint_dir) / "best_model.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_metrics": val_metrics,
                    "config": asdict(cfg),
                }, best_path)
                print(f"           -> saved new best checkpoint to {best_path}")
            else:
                epochs_without_improvement += 1

            # save resume state EVERY epoch (not just on improvement) so an interruption
            # never loses more than the current in-progress epoch
            save_resume_checkpoint(resume_path, epoch, model, optimizer,
                                    best_val_loss, epochs_without_improvement, cfg)

            if epochs_without_improvement >= cfg.early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1} "
                      f"(no val improvement for {cfg.early_stopping_patience} epochs)")
                break

    except KeyboardInterrupt:
        print(f"\nTraining interrupted. Progress through the last completed epoch is saved at "
              f"{resume_path}. Re-run with --resume true to continue from there.")
        return

    print(f"Training finished. Best val_loss={best_val_loss:.5f}")


if __name__ == "__main__":
    config = parse_args()
    print("Config:")
    for k, v in asdict(config).items():
        print(f"  {k}: {v}")
    train(config)