#!/usr/bin/env python3
"""Config-based training/evaluation entry point for RGB-to-HSI DifIISR.

Edit the CONFIG section and run:

    python main.py

This script intentionally has no command-line parser. It expects your
existing dataset loader at:

    dataset/dataset_loader.py

with:

    from dataset.dataset_loader import ARADDataset

The dataset may return either:
    (rgb, hsi)

or a dictionary containing:
    {"rgb": rgb, "hsi": hsi}

where rgb is [B, 3, H, W] and hsi is [B, NUM_BANDS, H, W].
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.dataset_loader import ARADDataset
from loss import mrae_loss, sam_loss, spectral_gradient_loss
from models import RGB2HSI_DifIISR


# ==================================================
# CONFIG
# ==================================================

MODE = "train"                 # "train" or "eval"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# Dataset configuration.
DATA_ROOT = "data"
HSI_KEY = "cube"
DOWNLOAD_DATA = True
TRAIN_IMAGES = 200
TOTAL_IMAGES = 230

# Dataloader.
BATCH_SIZE = 1
VAL_BATCH_SIZE = 1
NUM_WORKERS = 4
PIN_MEMORY = DEVICE == "cuda"

# Training.
NUM_EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 0.0
GRAD_CLIP_NORM = 1.0
USE_AMP = True

# Scheduler / early stopping.
EARLY_STOPPING_PATIENCE = 20
LR_PATIENCE = 5
LR_FACTOR = 0.5
MIN_LR = 1e-7

# Model.
NUM_BANDS = 31
BASE_CHANNELS = 32
DIFFUSION_STEPS = 15
KAPPA = 2.0
MIN_NOISE_LEVEL = 0.04
ETAS_END = 0.99
SCHEDULE_POWER = 0.3
PREDICT_TYPE = "xstart"        # "xstart", "epsilon", or "residual"
CONDITION_RGB = True
NUM_HEADS = 4
WINDOW_SIZE = 8
DROPOUT = 0.0

# Loss weights. These keys match RGB2HSI_DifIISR.compute_losses().
LOSS_WEIGHTS = {
    "diffusion": 1.0,
    "coarse_l1": 0.2,
    "recon_l1": 0.5,
    "mrae": 0.2,
    "sam": 0.05,
    "spectral_grad": 0.05,
    "rgb": 0.0,                 # set >0 only if RESPONSE_MATRIX_PATH is provided
}

# Optional spectral response matrix for RGB consistency.
# Expected shape: [3, NUM_BANDS]. Supported: .npy or torch checkpoint tensor.
RESPONSE_MATRIX_PATH: Optional[Union[str, Path]] = None

# If your HSI cubes are not normalized to [0, 1], set this to False.
CLIP_DENOISED = True

# Checkpoints.
CHECKPOINT_DIR = Path("checkpoints")
LATEST_PATH = CHECKPOINT_DIR / "rgb2hsi_difiisr_latest.pth"
BEST_PATH = CHECKPOINT_DIR / "rgb2hsi_difiisr_best.pth"
BEST_LOSS_PATH = CHECKPOINT_DIR / "rgb2hsi_difiisr_best_loss.pth"
RESUME_CHECKPOINT: Optional[Union[str, Path]] = None
EVAL_CHECKPOINT: Optional[Union[str, Path]] = None

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# REPRODUCIBILITY
# ==================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


# ==================================================
# SMALL UTILITIES
# ==================================================


def unpack_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize supported dataset outputs to rgb, hsi."""
    if isinstance(batch, dict):
        rgb = batch.get("rgb", batch.get("lq"))
        hsi = batch.get("hsi", batch.get("gt"))
        if rgb is None or hsi is None:
            raise KeyError(
                "Dictionary batch must contain ('rgb','hsi') or ('lq','gt'). "
                f"Available keys: {list(batch.keys())}"
            )
    elif isinstance(batch, (list, tuple)):
        if len(batch) < 2:
            raise ValueError("Tuple/list batch must contain at least [rgb, hsi].")
        rgb, hsi = batch[0], batch[1]
    else:
        raise TypeError(f"Unsupported batch type: {type(batch).__name__}")

    if not torch.is_tensor(rgb) or not torch.is_tensor(hsi):
        raise TypeError(
            "After DataLoader collation, RGB and HSI must be tensors. "
            f"Received rgb={type(rgb).__name__}, hsi={type(hsi).__name__}."
        )

    if rgb.ndim != 4 or hsi.ndim != 4:
        raise ValueError(
            "Expected batched tensors rgb=[B,3,H,W] and hsi=[B,L,H,W], "
            f"got rgb={tuple(rgb.shape)}, hsi={tuple(hsi.shape)}."
        )

    if rgb.shape[1] != 3:
        raise ValueError(f"Expected RGB to have 3 channels, got {rgb.shape[1]}.")

    if hsi.shape[1] != NUM_BANDS:
        raise ValueError(f"Expected HSI to have {NUM_BANDS} bands, got {hsi.shape[1]}.")

    return rgb, hsi


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(enabled: bool):
    try:
        return torch.amp.autocast("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)


def load_response_matrix(device: torch.device) -> Optional[torch.Tensor]:
    if RESPONSE_MATRIX_PATH is None:
        return None

    path = Path(RESPONSE_MATRIX_PATH)
    if not path.exists():
        raise FileNotFoundError(f"RESPONSE_MATRIX_PATH not found: {path}")

    if path.suffix.lower() == ".npy":
        matrix = torch.from_numpy(np.load(path)).float()
    else:
        loaded = torch.load(path, map_location="cpu")
        matrix = loaded.float() if torch.is_tensor(loaded) else torch.as_tensor(loaded).float()

    if matrix.shape != (3, NUM_BANDS):
        raise ValueError(
            f"Response matrix must have shape [3, {NUM_BANDS}], got {tuple(matrix.shape)}."
        )

    return matrix.to(device)


def model_config_dict() -> Dict[str, Any]:
    return {
        "bands": NUM_BANDS,
        "base_channels": BASE_CHANNELS,
        "steps": DIFFUSION_STEPS,
        "kappa": KAPPA,
        "min_noise_level": MIN_NOISE_LEVEL,
        "etas_end": ETAS_END,
        "schedule_power": SCHEDULE_POWER,
        "predict_type": PREDICT_TYPE,
        "condition_rgb": CONDITION_RGB,
        "num_heads": NUM_HEADS,
        "window_size": WINDOW_SIZE,
        "dropout": DROPOUT,
    }


def build_model(device: torch.device) -> RGB2HSI_DifIISR:
    model = RGB2HSI_DifIISR(**model_config_dict())
    return model.to(device)


# ==================================================
# DATA
# ==================================================


def make_dataloaders(device: torch.device) -> Tuple[Optional[DataLoader], DataLoader]:
    set_seed(SEED)
    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader: Optional[DataLoader] = None

    if MODE == "train":
        train_dataset = ARADDataset(
            root_dir=DATA_ROOT,
            train=True,
            train_images=TRAIN_IMAGES,
            total_images=TOTAL_IMAGES,
            cube_key=HSI_KEY,
            download=DOWNLOAD_DATA,
        )

        if len(train_dataset) == 0:
            raise RuntimeError("Training dataset is empty. Check DATA_ROOT and ARAD file pairing.")

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=(device.type == "cuda" and PIN_MEMORY),
            worker_init_fn=seed_worker if NUM_WORKERS > 0 else None,
            generator=generator,
            drop_last=False,
        )

    val_dataset = ARADDataset(
        root_dir=DATA_ROOT,
        train=False,
        train_images=TRAIN_IMAGES,
        total_images=TOTAL_IMAGES,
        cube_key=HSI_KEY,
        download=DOWNLOAD_DATA if MODE == "eval" else False,
    )

    if len(val_dataset) == 0:
        raise RuntimeError("Validation dataset is empty. Check TRAIN_IMAGES/TOTAL_IMAGES and file pairing.")

    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda" and PIN_MEMORY),
        worker_init_fn=seed_worker if NUM_WORKERS > 0 else None,
        drop_last=False,
    )

    return train_loader, val_loader


# ==================================================
# METRICS
# ==================================================


def rmse_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(F.mse_loss(pred, target).clamp_min(1e-12))


def psnr_metric(pred: torch.Tensor, target: torch.Tensor, max_value: float = 1.0) -> torch.Tensor:
    mse = F.mse_loss(pred, target).clamp_min(1e-12)
    max_tensor = torch.tensor(max_value, device=pred.device, dtype=pred.dtype)
    return 20.0 * torch.log10(max_tensor) - 10.0 * torch.log10(mse)


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    return {
        "mrae": float(mrae_loss(pred, target).item()),
        "rmse": float(rmse_metric(pred, target).item()),
        "sam": float(sam_loss(pred, target).item()),
        "psnr": float(psnr_metric(pred, target).item()),
    }


# ==================================================
# CHECKPOINTS
# ==================================================


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau],
    best_val_mrae: float,
    best_val_loss: float,
    epochs_without_improvement: int,
) -> None:
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "model_config": model_config_dict(),
        "loss_weights": LOSS_WEIGHTS,
        "best_val_mrae": best_val_mrae,
        "best_val_loss": best_val_loss,
        "epochs_without_improvement": epochs_without_improvement,
    }

    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()

    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Union[str, Path], device: torch.device) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(
            f"{path} is not a valid RGB-to-HSI DifIISR checkpoint. "
            "Expected a dictionary containing a 'model' state dict."
        )

    return checkpoint


# ==================================================
# VALIDATION
# ==================================================


@torch.no_grad()
def validate(
    model: RGB2HSI_DifIISR,
    val_loader: DataLoader,
    device: torch.device,
    response_matrix: Optional[torch.Tensor],
) -> Dict[str, float]:
    model.eval()

    totals = {
        "loss": 0.0,
        "mrae": 0.0,
        "rmse": 0.0,
        "sam": 0.0,
        "psnr": 0.0,
    }
    count = 0

    for batch in val_loader:
        rgb, hsi = unpack_batch(batch)
        rgb = rgb.to(device, non_blocking=(device.type == "cuda"))
        hsi = hsi.to(device, non_blocking=(device.type == "cuda"))

        sample_out = model.sample(
            rgb=rgb,
            clip_denoised=CLIP_DENOISED,
            return_all=False,
        )
        pred_hsi = sample_out["hsi"]

        val_loss = (
            LOSS_WEIGHTS["recon_l1"] * F.l1_loss(pred_hsi, hsi)
            + LOSS_WEIGHTS["mrae"] * mrae_loss(pred_hsi, hsi)
            + LOSS_WEIGHTS["sam"] * sam_loss(pred_hsi, hsi)
            + LOSS_WEIGHTS["spectral_grad"] * spectral_gradient_loss(pred_hsi, hsi)
        )

        # Optional RGB consistency for validation if a response matrix is available.
        if response_matrix is not None and LOSS_WEIGHTS.get("rgb", 0.0) > 0:
            val_forward = model(
                rgb=rgb,
                hsi_gt=hsi,
                return_loss=True,
                response_matrix=response_matrix,
                loss_weights=LOSS_WEIGHTS,
            )
            val_loss = val_forward["loss"]

        batch_metrics = compute_metrics(pred_hsi, hsi)
        batch_size = rgb.shape[0]
        totals["loss"] += float(val_loss.item()) * batch_size
        for name in ("mrae", "rmse", "sam", "psnr"):
            totals[name] += batch_metrics[name] * batch_size
        count += batch_size

    if count == 0:
        raise RuntimeError("Validation loader is empty.")

    return {name: value / count for name, value in totals.items()}


# ==================================================
# TRAINING
# ==================================================


def train() -> None:
    set_seed(SEED)

    device = torch.device(DEVICE)
    response_matrix = load_response_matrix(device)
    train_loader, val_loader = make_dataloaders(device)

    if train_loader is None:
        raise RuntimeError("MODE='train' but train_loader is None.")

    model = build_model(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.99),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=MIN_LR,
    )

    amp_enabled = USE_AMP and device.type == "cuda"
    scaler = make_grad_scaler(amp_enabled)

    start_epoch = 1
    best_val_mrae = math.inf
    best_val_loss = math.inf
    epochs_without_improvement = 0

    if RESUME_CHECKPOINT is not None:
        checkpoint = load_checkpoint(RESUME_CHECKPOINT, device)
        saved_config = checkpoint.get("model_config", None)
        if saved_config is not None and saved_config != model_config_dict():
            raise ValueError(
                "Resume checkpoint architecture differs from current CONFIG. "
                "Either update CONFIG or set RESUME_CHECKPOINT=None."
            )
        model.load_state_dict(checkpoint["model"], strict=True)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val_mrae = float(checkpoint.get("best_val_mrae", math.inf))
        best_val_loss = float(checkpoint.get("best_val_loss", math.inf))
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        print(f"Resumed from epoch {start_epoch}: {RESUME_CHECKPOINT}")

    print(f"Device: {device}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Model config: {model_config_dict()}")
    print(f"Loss weights: {LOSS_WEIGHTS}")

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        model.train()

        running = {
            "loss": 0.0,
            "diffusion": 0.0,
            "coarse_l1": 0.0,
            "recon_l1": 0.0,
            "mrae": 0.0,
            "sam": 0.0,
            "spectral_grad": 0.0,
        }
        count = 0

        for batch in train_loader:
            rgb, hsi = unpack_batch(batch)
            rgb = rgb.to(device, non_blocking=(device.type == "cuda"))
            hsi = hsi.to(device, non_blocking=(device.type == "cuda"))

            optimizer.zero_grad(set_to_none=True)

            with autocast_context(amp_enabled):
                out = model(
                    rgb=rgb,
                    hsi_gt=hsi,
                    return_loss=True,
                    response_matrix=response_matrix,
                    loss_weights=LOSS_WEIGHTS,
                )
                loss = out["loss"]

            scaler.scale(loss).backward()

            if GRAD_CLIP_NORM > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)

            scaler.step(optimizer)
            scaler.update()

            batch_size = rgb.shape[0]
            running["loss"] += float(out["loss"].item()) * batch_size
            running["diffusion"] += float(out["loss_diffusion"].item()) * batch_size
            running["coarse_l1"] += float(out["loss_coarse_l1"].item()) * batch_size
            running["recon_l1"] += float(out["loss_recon_l1"].item()) * batch_size
            running["mrae"] += float(out["loss_mrae"].item()) * batch_size
            running["sam"] += float(out["loss_sam"].item()) * batch_size
            running["spectral_grad"] += float(out["loss_spectral_grad"].item()) * batch_size
            count += batch_size

        train_loss = running["loss"] / max(count, 1)
        train_diffusion = running["diffusion"] / max(count, 1)
        train_mrae = running["mrae"] / max(count, 1)
        train_sam = running["sam"] / max(count, 1)

        val_results = validate(model, val_loader, device, response_matrix)
        scheduler.step(val_results["mrae"])
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{NUM_EPOCHS} "
            f"| Train Loss {train_loss:.6f} "
            f"| Train Diff {train_diffusion:.6f} "
            f"| Train MRAE {train_mrae:.6f} "
            f"| Train SAM {train_sam:.4f} "
            f"| Val Loss {val_results['loss']:.6f} "
            f"| Val MRAE {val_results['mrae']:.6f} "
            f"| Val RMSE {val_results['rmse']:.6f} "
            f"| Val SAM {val_results['sam']:.4f} "
            f"| Val PSNR {val_results['psnr']:.4f} "
            f"| LR {current_lr:.2e}"
        )

        if val_results["loss"] < best_val_loss:
            best_val_loss = val_results["loss"]
            save_checkpoint(
                BEST_LOSS_PATH,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_mrae=best_val_mrae,
                best_val_loss=best_val_loss,
                epochs_without_improvement=epochs_without_improvement,
            )

        if val_results["mrae"] < best_val_mrae:
            best_val_mrae = val_results["mrae"]
            epochs_without_improvement = 0
            save_checkpoint(
                BEST_PATH,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_mrae=best_val_mrae,
                best_val_loss=best_val_loss,
                epochs_without_improvement=epochs_without_improvement,
            )
            print(f"Saved best model with Val MRAE {best_val_mrae:.6f}")
        else:
            epochs_without_improvement += 1
            print(
                f"No Val MRAE improvement for "
                f"{epochs_without_improvement}/{EARLY_STOPPING_PATIENCE} epochs"
            )

        save_checkpoint(
            LATEST_PATH,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_val_mrae=best_val_mrae,
            best_val_loss=best_val_loss,
            epochs_without_improvement=epochs_without_improvement,
        )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping. Best Val MRAE: {best_val_mrae:.6f}")
            break


# ==================================================
# EVALUATION
# ==================================================


def evaluate() -> None:
    set_seed(SEED)

    device = torch.device(DEVICE)
    response_matrix = load_response_matrix(device)
    _, val_loader = make_dataloaders(device)

    checkpoint_path = Path(EVAL_CHECKPOINT) if EVAL_CHECKPOINT is not None else BEST_PATH
    checkpoint = load_checkpoint(checkpoint_path, device)

    saved_config = checkpoint.get("model_config", None)
    if saved_config is not None and saved_config != model_config_dict():
        raise ValueError(
            "Evaluation checkpoint architecture differs from current CONFIG. "
            "Set CONFIG to match the checkpoint or use the correct checkpoint."
        )

    model = build_model(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    val_results = validate(model, val_loader, device, response_matrix)

    print(f"Evaluated checkpoint: {checkpoint_path}")
    print(
        f"MRAE {val_results['mrae']:.6f} "
        f"| RMSE {val_results['rmse']:.6f} "
        f"| SAM {val_results['sam']:.4f} "
        f"| PSNR {val_results['psnr']:.4f}"
    )


# ==================================================
# MAIN
# ==================================================


def main() -> None:
    if MODE == "train":
        train()
    elif MODE == "eval":
        evaluate()
    else:
        raise ValueError("MODE must be 'train' or 'eval'.")


if __name__ == "__main__":
    main()
