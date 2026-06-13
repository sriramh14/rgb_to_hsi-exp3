"""
Train RGB2HSI_DifIISR on ARAD/NTIRE2020 RGB-HSI pairs.

Expected default dataset layout after download:
    data/
      NTIRE2020_Train_Spectral/*.mat
      NTIRE2020_Train_RealWorld/*.jpg

The ARADDataset class below is adapted from the loader provided by the user.
It returns:
    rgb: [3, image_size, image_size], float32
    hsi: [bands, image_size, image_size], float32

Example:
    python main.py --root_dir data --download --epochs 100 --batch_size 1 --base_channels 32

Resume:
    python main.py --resume exp_rgb2hsi/latest.pth
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.io as sio
from PIL import Image

import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from models import RGB2HSI_DifIISR


class ARADDataset(Dataset):
    """
    ARAD / NTIRE2020 paired RGB-HSI dataset loader.

    This follows the loader provided in the chat, with a few small additions:
      - configurable image_size;
      - optional HSI normalization mode;
      - safer HSI channel layout handling;
      - optional bands check.
    """

    def __init__(
        self,
        root_dir: str = "data",
        train: bool = True,
        train_images: int = 200,
        total_images: int = 230,
        cube_key: str = "cube",
        download: bool = True,
        image_size: int = 256,
        bands: Optional[int] = 31,
        hsi_norm: str = "none",
        hsi_scale: float = 1.0,
    ):
        super().__init__()
        if hsi_norm not in {"none", "sample_max", "constant"}:
            raise ValueError("hsi_norm must be one of: none, sample_max, constant")

        self.cube_key = cube_key
        self.image_size = int(image_size)
        self.bands = bands
        self.hsi_norm = hsi_norm
        self.hsi_scale = float(hsi_scale)

        spectral_dir = os.path.join(root_dir, "NTIRE2020_Train_Spectral")
        rgb_dir = os.path.join(root_dir, "NTIRE2020_Train_RealWorld")

        os.makedirs(spectral_dir, exist_ok=True)
        os.makedirs(rgb_dir, exist_ok=True)

        if download:
            self._download_if_needed(
                root_dir=root_dir,
                spectral_dir=spectral_dir,
                rgb_dir=rgb_dir,
                total_images=total_images,
            )

        hsi_files = sorted([f for f in os.listdir(spectral_dir) if f.endswith(".mat")])[:total_images]

        rgb_lookup = {
            f.replace("_RealWorld.jpg", ""): f
            for f in os.listdir(rgb_dir)
            if f.endswith(".jpg")
        }

        pairs = []
        for hsi_name in hsi_files:
            stem = os.path.splitext(hsi_name)[0]
            if stem not in rgb_lookup:
                continue
            pairs.append((os.path.join(spectral_dir, hsi_name), os.path.join(rgb_dir, rgb_lookup[stem])))

        print(f"Found {len(pairs)} paired samples")

        if train:
            self.pairs = pairs[:train_images]
        else:
            self.pairs = pairs[train_images:]

        split_name = "Train" if train else "Val"
        print(f"{split_name}: {len(self.pairs)} samples")

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No samples found for {split_name}. Check root_dir, train_images, total_images, and filenames."
            )

    @staticmethod
    def _download_if_needed(root_dir: str, spectral_dir: str, rgb_dir: str, total_images: int) -> None:
        existing_hsi = [f for f in os.listdir(spectral_dir) if f.endswith(".mat")]
        existing_rgb = [f for f in os.listdir(rgb_dir) if f.endswith(".jpg")]

        if len(existing_hsi) >= total_images and len(existing_rgb) >= total_images:
            return

        from huggingface_hub import hf_hub_download, list_repo_files

        print(f"Downloading {total_images} HSI files and {total_images} RGB files...")

        repo_files = list_repo_files("mhmdjouni/arad_hsdb", repo_type="dataset")

        hsi_files = sorted(
            [f for f in repo_files if f.endswith(".mat") and "NTIRE2020_Train_Spectral" in f]
        )[:total_images]

        rgb_files = sorted(
            [f for f in repo_files if f.endswith(".jpg") and "NTIRE2020_Train_RealWorld" in f]
        )[:total_images]

        for file in hsi_files:
            hf_hub_download(
                repo_id="mhmdjouni/arad_hsdb",
                repo_type="dataset",
                filename=file,
                local_dir=root_dir,
                local_dir_use_symlinks=False,
            )

        for file in rgb_files:
            hf_hub_download(
                repo_id="mhmdjouni/arad_hsdb",
                repo_type="dataset",
                filename=file,
                local_dir=root_dir,
                local_dir_use_symlinks=False,
            )

        print("Download complete")

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_hsi(self, hsi_path: str) -> torch.Tensor:
        mat = sio.loadmat(hsi_path)
        if self.cube_key not in mat:
            keys = [k for k in mat.keys() if not k.startswith("__")]
            raise KeyError(f"cube_key='{self.cube_key}' not found in {hsi_path}. Available keys: {keys}")

        hsi = mat[self.cube_key].astype(np.float32)
        if hsi.ndim != 3:
            raise ValueError(f"Expected 3D HSI cube, got shape {hsi.shape} in {hsi_path}")

        # ARAD/NTIRE cubes are usually [H, W, C]. If already [C, H, W], keep it.
        if self.bands is not None:
            if hsi.shape[-1] == self.bands:
                hsi = np.transpose(hsi, (2, 0, 1))
            elif hsi.shape[0] == self.bands:
                pass
            else:
                raise ValueError(
                    f"Could not identify spectral dimension for {hsi_path}. "
                    f"Expected {self.bands} bands, got shape {hsi.shape}."
                )
        else:
            # Fallback to the user's original assumption: [H,W,C] -> [C,H,W].
            hsi = np.transpose(hsi, (2, 0, 1))

        if self.hsi_norm == "sample_max":
            max_val = float(np.max(hsi))
            if max_val > 0:
                hsi = hsi / max_val
        elif self.hsi_norm == "constant":
            hsi = hsi / max(self.hsi_scale, 1e-12)
        # hsi_norm == "none" follows the user-provided loader exactly.

        hsi_t = torch.from_numpy(hsi).float()
        hsi_t = F.interpolate(
            hsi_t.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return hsi_t

    def _load_rgb(self, rgb_path: str) -> torch.Tensor:
        rgb = Image.open(rgb_path).convert("RGB")
        rgb = np.array(rgb, dtype=np.float32) / 255.0
        rgb = np.transpose(rgb, (2, 0, 1))
        rgb_t = torch.from_numpy(rgb).float()
        rgb_t = F.interpolate(
            rgb_t.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return rgb_t

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        hsi_path, rgb_path = self.pairs[idx]
        hsi = self._load_hsi(hsi_path)
        rgb = self._load_rgb(rgb_path)
        return rgb, hsi


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=(num_workers > 0),
    )


def load_response_matrix(path: Optional[str], bands: int, device: torch.device) -> Optional[torch.Tensor]:
    if path is None or path == "":
        return None

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"response_matrix not found: {path}")

    if path_obj.suffix.lower() == ".npy":
        matrix = np.load(path_obj)
    elif path_obj.suffix.lower() == ".npz":
        loaded = np.load(path_obj)
        key = "response" if "response" in loaded else list(loaded.keys())[0]
        matrix = loaded[key]
    elif path_obj.suffix.lower() == ".mat":
        loaded = sio.loadmat(path_obj)
        keys = [k for k in loaded.keys() if not k.startswith("__")]
        key = "response" if "response" in loaded else keys[0]
        matrix = loaded[key]
    elif path_obj.suffix.lower() in {".csv", ".txt"}:
        matrix = np.loadtxt(path_obj, delimiter="," if path_obj.suffix.lower() == ".csv" else None)
    else:
        raise ValueError("response_matrix must be .npy, .npz, .mat, .csv, or .txt")

    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape == (bands, 3):
        matrix = matrix.T
    if matrix.shape != (3, bands):
        raise ValueError(f"Expected response matrix shape [3,{bands}] or [{bands},3], got {matrix.shape}")

    matrix_t = torch.from_numpy(matrix).float().to(device)
    return matrix_t


def loss_weight_dict(args: argparse.Namespace) -> Dict[str, float]:
    return {
        "diffusion": args.w_diffusion,
        "coarse_l1": args.w_coarse_l1,
        "recon_l1": args.w_recon_l1,
        "mrae": args.w_mrae,
        "sam": args.w_sam,
        "spectral_grad": args.w_spectral_grad,
        "rgb": args.w_rgb,
    }


def sam_metric(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred_f = pred.flatten(2)
    target_f = target.flatten(2)
    dot = (pred_f * target_f).sum(dim=1)
    denom = torch.linalg.norm(pred_f, dim=1) * torch.linalg.norm(target_f, dim=1) + eps
    angle = torch.acos(torch.clamp(dot / denom, -1.0 + 1e-6, 1.0 - 1e-6))
    return angle.mean()


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> Dict[str, float]:
    pred = pred.float()
    target = target.float()
    mse = F.mse_loss(pred, target)
    rmse = torch.sqrt(mse)
    mrae = ((pred - target).abs() / (target.abs() + 1e-3)).mean()
    psnr = 20.0 * torch.log10(torch.tensor(float(data_range), device=pred.device) / rmse.clamp_min(1e-12))
    sam = sam_metric(pred, target)
    return {
        "mrae": float(mrae.detach().cpu()),
        "rmse": float(rmse.detach().cpu()),
        "psnr": float(psnr.detach().cpu()),
        "sam": float(sam.detach().cpu()),
    }


class AverageMeter:
    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    best_mrae: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_mrae": best_mrae,
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[GradScaler] = None,
    device: Optional[torch.device] = None,
) -> Tuple[int, float]:
    ckpt = torch.load(path, map_location=device or "cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_mrae = float(ckpt.get("best_mrae", math.inf))
    return start_epoch, best_mrae


def train_one_epoch(
    model: RGB2HSI_DifIISR,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    args: argparse.Namespace,
    response_matrix: Optional[torch.Tensor],
    epoch: int,
) -> Dict[str, float]:
    model.train()
    weights = loss_weight_dict(args)
    meters = {name: AverageMeter() for name in [
        "loss", "loss_diffusion", "loss_coarse_l1", "loss_recon_l1",
        "loss_mrae", "loss_sam", "loss_spectral_grad", "loss_rgb"
    ]}

    start = time.time()
    optimizer.zero_grad(set_to_none=True)

    for step, (rgb, hsi) in enumerate(loader, start=1):
        rgb = rgb.to(device, non_blocking=True)
        hsi = hsi.to(device, non_blocking=True)

        with autocast(enabled=args.amp and device.type == "cuda"):
            out = model(
                rgb=rgb,
                hsi_gt=hsi,
                return_loss=True,
                response_matrix=response_matrix,
                loss_weights=weights,
            )
            loss = out["loss"] / args.grad_accum_steps

        scaler.scale(loss).backward()

        if step % args.grad_accum_steps == 0:
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        batch_size = rgb.shape[0]
        for name in meters.keys():
            if name in out:
                meters[name].update(float(out[name].detach().cpu()), batch_size)

        if step % args.log_every == 0:
            elapsed = time.time() - start
            print(
                f"Epoch {epoch:03d} | Step {step:04d}/{len(loader)} | "
                f"Loss {meters['loss'].avg:.5f} | "
                f"MRAE-loss {meters['loss_mrae'].avg:.5f} | "
                f"SAM-loss {meters['loss_sam'].avg:.5f} | "
                f"Time {elapsed:.1f}s"
            )

    return {name: meter.avg for name, meter in meters.items()}


@torch.no_grad()
def validate(
    model: RGB2HSI_DifIISR,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    response_matrix: Optional[torch.Tensor],
) -> Dict[str, float]:
    model.eval()
    weights = loss_weight_dict(args)
    loss_meter = AverageMeter()
    metric_meters = {name: AverageMeter() for name in ["mrae", "rmse", "psnr", "sam"]}

    for rgb, hsi in loader:
        rgb = rgb.to(device, non_blocking=True)
        hsi = hsi.to(device, non_blocking=True)

        if args.val_mode == "sample":
            sample_out = model.sample(rgb, clip_denoised=args.clip_val_pred)
            pred = sample_out["hsi"]
            # Also compute training loss on a deterministic low-noise step for logging.
            t = torch.zeros((rgb.shape[0],), device=device, dtype=torch.long)
            noise = torch.zeros_like(hsi)
            out = model(
                rgb=rgb,
                hsi_gt=hsi,
                t=t,
                noise=noise,
                return_loss=True,
                response_matrix=response_matrix,
                loss_weights=weights,
            )
        else:
            # Fast validation: denoise the lowest-noise state with zero injected noise.
            t = torch.zeros((rgb.shape[0],), device=device, dtype=torch.long)
            noise = torch.zeros_like(hsi)
            out = model(
                rgb=rgb,
                hsi_gt=hsi,
                t=t,
                noise=noise,
                return_loss=True,
                response_matrix=response_matrix,
                loss_weights=weights,
            )
            pred = out["pred_hsi"]
            if args.clip_val_pred:
                pred = pred.clamp(0, 1)

        batch_size = rgb.shape[0]
        loss_meter.update(float(out["loss"].detach().cpu()), batch_size)
        metrics = compute_metrics(pred, hsi, data_range=args.metric_data_range)
        for name, value in metrics.items():
            metric_meters[name].update(value, batch_size)

    results = {f"val_{name}": meter.avg for name, meter in metric_meters.items()}
    results["val_loss"] = loss_meter.avg
    return results


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train DifIISR-style RGB-to-HSI residual diffusion model")

    # Dataset
    parser.add_argument("--root_dir", type=str, default="data")
    parser.add_argument("--download", action="store_true", help="Download ARAD/NTIRE2020 files from Hugging Face if missing")
    parser.add_argument("--no_download", action="store_true", help="Disable download even if files are missing")
    parser.add_argument("--train_images", type=int, default=200)
    parser.add_argument("--total_images", type=int, default=230)
    parser.add_argument("--cube_key", type=str, default="cube")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--bands", type=int, default=31)
    parser.add_argument("--hsi_norm", type=str, default="none", choices=["none", "sample_max", "constant"])
    parser.add_argument("--hsi_scale", type=float, default=1.0)

    # Model
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--kappa", type=float, default=2.0)
    parser.add_argument("--min_noise_level", type=float, default=0.04)
    parser.add_argument("--etas_end", type=float, default=0.99)
    parser.add_argument("--schedule_power", type=float, default=0.3)
    parser.add_argument("--predict_type", type=str, default="xstart", choices=["xstart", "epsilon", "residual"])
    parser.add_argument("--condition_rgb", action="store_true", default=True)
    parser.add_argument("--no_condition_rgb", action="store_false", dest="condition_rgb")
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)

    # Loss weights
    parser.add_argument("--w_diffusion", type=float, default=1.0)
    parser.add_argument("--w_coarse_l1", type=float, default=0.2)
    parser.add_argument("--w_recon_l1", type=float, default=0.5)
    parser.add_argument("--w_mrae", type=float, default=0.2)
    parser.add_argument("--w_sam", type=float, default=0.05)
    parser.add_argument("--w_spectral_grad", type=float, default=0.05)
    parser.add_argument("--w_rgb", type=float, default=0.0)
    parser.add_argument("--response_matrix", type=str, default=None)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA")
    parser.add_argument("--seed", type=int, default=1234)

    # Validation / logging
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--val_mode", type=str, default="denoise", choices=["denoise", "sample"])
    parser.add_argument("--clip_val_pred", action="store_true", default=True)
    parser.add_argument("--no_clip_val_pred", action="store_false", dest="clip_val_pred")
    parser.add_argument("--metric_data_range", type=float, default=1.0)

    # I/O
    parser.add_argument("--out_dir", type=str, default="exp_rgb2hsi")
    parser.add_argument("--resume", type=str, default=None)

    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    # --no_download has priority over --download.
    download = bool(args.download and not args.no_download)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    train_set = ARADDataset(
        root_dir=args.root_dir,
        train=True,
        train_images=args.train_images,
        total_images=args.total_images,
        cube_key=args.cube_key,
        download=download,
        image_size=args.image_size,
        bands=args.bands,
        hsi_norm=args.hsi_norm,
        hsi_scale=args.hsi_scale,
    )
    val_set = ARADDataset(
        root_dir=args.root_dir,
        train=False,
        train_images=args.train_images,
        total_images=args.total_images,
        cube_key=args.cube_key,
        download=False,
        image_size=args.image_size,
        bands=args.bands,
        hsi_norm=args.hsi_norm,
        hsi_scale=args.hsi_scale,
    )

    train_loader = make_loader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers, seed=args.seed)
    val_loader = make_loader(val_set, args.batch_size, shuffle=False, num_workers=args.num_workers, seed=args.seed)

    model = RGB2HSI_DifIISR(
        bands=args.bands,
        base_channels=args.base_channels,
        steps=args.steps,
        kappa=args.kappa,
        min_noise_level=args.min_noise_level,
        etas_end=args.etas_end,
        schedule_power=args.schedule_power,
        predict_type=args.predict_type,
        condition_rgb=args.condition_rgb,
        num_heads=args.num_heads,
        window_size=args.window_size,
        dropout=args.dropout,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params / 1e6:.2f} M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")
    response_matrix = load_response_matrix(args.response_matrix, bands=args.bands, device=device)

    start_epoch = 1
    best_mrae = math.inf
    if args.resume:
        start_epoch, best_mrae = load_checkpoint(args.resume, model, optimizer, scaler, device)
        print(f"Resumed from {args.resume}: start_epoch={start_epoch}, best_mrae={best_mrae:.6f}")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            args=args,
            response_matrix=response_matrix,
            epoch=epoch,
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"Train Loss {train_stats['loss']:.6f} | "
            f"Diff {train_stats['loss_diffusion']:.6f} | "
            f"MRAE-loss {train_stats['loss_mrae']:.6f} | "
            f"SAM-loss {train_stats['loss_sam']:.6f}"
        )

        did_val = (epoch % args.val_every == 0) or (epoch == args.epochs)
        if did_val:
            val_stats = validate(model, val_loader, device, args, response_matrix)
            print(
                f"Epoch {epoch:03d}/{args.epochs:03d} | "
                f"Val Loss {val_stats['val_loss']:.6f} | "
                f"Val MRAE {val_stats['val_mrae']:.6f} | "
                f"Val RMSE {val_stats['val_rmse']:.6f} | "
                f"Val SAM {val_stats['val_sam']:.6f} | "
                f"Val PSNR {val_stats['val_psnr']:.3f}"
            )

            if val_stats["val_mrae"] < best_mrae:
                best_mrae = val_stats["val_mrae"]
                save_checkpoint(out_dir / "best.pth", model, optimizer, scaler, epoch, best_mrae, args)
                print(f"Saved best checkpoint: {out_dir / 'best.pth'}")

        save_checkpoint(out_dir / "latest.pth", model, optimizer, scaler, epoch, best_mrae, args)
        if epoch % args.save_every == 0:
            save_checkpoint(out_dir / f"epoch_{epoch:03d}.pth", model, optimizer, scaler, epoch, best_mrae, args)

        print(f"Epoch time: {time.time() - epoch_start:.1f}s | Best Val MRAE: {best_mrae:.6f}")


if __name__ == "__main__":
    main()
