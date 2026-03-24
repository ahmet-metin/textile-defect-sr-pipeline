import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image
from ultralytics import YOLO

# Project imports
# Assumes this file lives at: src/edge/rpi_demo.py
# and the repo uses the modular layout we discussed.
from src.esrgan.model import Generator
from src.esrgan.dataset import build_transforms
from src.esrgan.utils import load_checkpoint


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_images(input_dir: str | Path) -> list[Path]:
    input_dir = Path(input_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted([p for p in input_dir.iterdir() if p.suffix.lower() in exts])


def tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert model output tensor [1, C, H, W] or [C, H, W] to uint8 RGB image.
    """
    if tensor.ndim == 4:
        tensor = tensor[0]
    img = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    img = (img * 255.0).round().astype(np.uint8)
    return img


def upscale_with_esrgan(
    image_bgr: np.ndarray,
    gen: Generator,
    device: str,
    high_res: int = 416,
) -> tuple[np.ndarray, float]:
    """
    Takes a BGR image, resizes it to the model input size, applies ESRGAN, and returns
    an RGB upscaled image.
    """
    _, _, _, test_transform = build_transforms(high_res)

    image_bgr = cv2.resize(image_bgr, (high_res, high_res), interpolation=cv2.INTER_AREA)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    input_tensor = test_transform(image=image_rgb)["image"].unsqueeze(0).to(device)

    start = time.time()
    with torch.no_grad():
        sr_tensor = gen(input_tensor)
    elapsed = time.time() - start

    sr_rgb = tensor_to_uint8_image(sr_tensor)
    return sr_rgb, elapsed


def split_into_tiles(image_rgb: np.ndarray, tile_size: int = 416) -> tuple[list[np.ndarray], list[tuple[int, int]], tuple[int, int]]:
    """
    Split an image into non-overlapping tiles. Pads the image if needed.
    Returns:
        tiles, positions, padded_shape
    where positions are (y, x) top-left corners.
    """
    h, w, c = image_rgb.shape

    pad_h = (tile_size - (h % tile_size)) % tile_size
    pad_w = (tile_size - (w % tile_size)) % tile_size

    padded = cv2.copyMakeBorder(
        image_rgb,
        0,
        pad_h,
        0,
        pad_w,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    ph, pw, _ = padded.shape
    tiles = []
    positions = []

    for y in range(0, ph, tile_size):
        for x in range(0, pw, tile_size):
            tile = padded[y:y + tile_size, x:x + tile_size].copy()
            tiles.append(tile)
            positions.append((y, x))

    return tiles, positions, (ph, pw)


def reconstruct_from_tiles(
    tile_images: list[np.ndarray],
    positions: list[tuple[int, int]],
    padded_shape: tuple[int, int],
    original_shape: tuple[int, int],
    tile_size: int = 416,
) -> np.ndarray:
    """
    Reconstruct full image from processed tiles and crop to the original shape.
    """
    ph, pw = padded_shape
    oh, ow = original_shape
    canvas = np.zeros((ph, pw, 3), dtype=np.uint8)

    for tile_img, (y, x) in zip(tile_images, positions):
        canvas[y:y + tile_size, x:x + tile_size] = tile_img

    return canvas[:oh, :ow]


def run_yolo_on_tile(tile_rgb: np.ndarray, yolo_model: YOLO, conf: float = 0.25) -> np.ndarray:
    """
    Run YOLO on a single tile and return the plotted RGB result.
    """
    results = yolo_model.predict(source=tile_rgb, conf=conf, verbose=False)
    plotted = results[0].plot()  # usually BGR-like ndarray from Ultralytics
    if plotted.ndim == 3 and plotted.shape[2] == 3:
        plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
    return plotted


def process_single_image(
    image_bgr: np.ndarray,
    gen: Generator,
    yolo_model: YOLO,
    device: str,
    out_dir: str | Path,
    stem: str,
    input_size: int = 416,
    tile_size: int = 416,
    save_intermediate: bool = True,
) -> None:
    out_dir = ensure_dir(out_dir)
    sr_dir = ensure_dir(out_dir / "sr")
    tiles_dir = ensure_dir(out_dir / "tiles")
    pred_tiles_dir = ensure_dir(out_dir / "pred_tiles")
    recon_dir = ensure_dir(out_dir / "reconstructed")

    # 1) ESRGAN super-resolution
    sr_rgb, sr_elapsed = upscale_with_esrgan(
        image_bgr=image_bgr,
        gen=gen,
        device=device,
        high_res=input_size,
    )

    print(f"[{stem}] ESRGAN time: {sr_elapsed:.4f} sec")

    if save_intermediate:
        Image.fromarray(sr_rgb).save(sr_dir / f"{stem}_sr.png")

    # 2) Split into tiles
    original_h, original_w = sr_rgb.shape[:2]
    tiles, positions, padded_shape = split_into_tiles(sr_rgb, tile_size=tile_size)

    # 3) Run YOLO tile by tile
    pred_tiles = []
    yolo_total = 0.0

    for idx, (tile, (y, x)) in enumerate(zip(tiles, positions)):
        if save_intermediate:
            Image.fromarray(tile).save(tiles_dir / f"{stem}_tile_{idx:02d}_{y}_{x}.png")

        t0 = time.time()
        pred_tile = run_yolo_on_tile(tile, yolo_model)
        yolo_total += time.time() - t0

        pred_tiles.append(pred_tile)

        if save_intermediate:
            Image.fromarray(pred_tile).save(pred_tiles_dir / f"{stem}_pred_{idx:02d}_{y}_{x}.png")

    print(f"[{stem}] YOLO tile inference total: {yolo_total:.4f} sec")
    print(f"[{stem}] Combined pipeline total: {sr_elapsed + yolo_total:.4f} sec")

    # 4) Reconstruct full predicted image
    reconstructed = reconstruct_from_tiles(
        tile_images=pred_tiles,
        positions=positions,
        padded_shape=padded_shape,
        original_shape=(original_h, original_w),
        tile_size=tile_size,
    )

    Image.fromarray(reconstructed).save(recon_dir / f"{stem}_reconstructed.png")


def load_esrgan_generator(checkpoint_path: str, device: str) -> Generator:
    """
    Load a lightweight ESRGAN generator for Raspberry Pi deployment.
    The original RPi code reduced channels and RRDB blocks for feasibility.
    """
    gen = Generator(
        in_channels=3,
        num_channels=16,  # lightweight version for RPi
        num_blocks=1,     # lightweight version for RPi
    ).to(device)

    opt_dummy = torch.optim.Adam(gen.parameters(), lr=1e-4)
    load_checkpoint(checkpoint_path, gen, opt_dummy, lr=1e-4, device=device)
    gen.eval()
    return gen


def process_from_folder(
    input_dir: str,
    output_dir: str,
    esrgan_weights: str,
    yolo_weights: str,
    input_size: int,
    tile_size: int,
    conf: float,
    device: str,
) -> None:
    gen = load_esrgan_generator(esrgan_weights, device=device)
    yolo_model = YOLO(yolo_weights)

    image_paths = list_images(input_dir)
    if not image_paths:
        print(f"No images found in: {input_dir}")
        return

    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        process_single_image(
            image_bgr=image_bgr,
            gen=gen,
            yolo_model=yolo_model,
            device=device,
            out_dir=output_dir,
            stem=image_path.stem,
            input_size=input_size,
            tile_size=tile_size,
            save_intermediate=True,
        )


def process_from_camera(
    camera_index: int,
    output_dir: str,
    esrgan_weights: str,
    yolo_weights: str,
    input_size: int,
    tile_size: int,
    conf: float,
    device: str,
    max_frames: int = 1,
) -> None:
    gen = load_esrgan_generator(esrgan_weights, device=device)
    yolo_model = YOLO(yolo_weights)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    frame_count = 0
    try:
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera.")
                break

            stem = f"camera_frame_{frame_count:04d}"
            process_single_image(
                image_bgr=frame,
                gen=gen,
                yolo_model=yolo_model,
                device=device,
                out_dir=output_dir,
                stem=stem,
                input_size=input_size,
                tile_size=tile_size,
                save_intermediate=True,
            )
            frame_count += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Raspberry Pi demo: ESRGAN -> tiling -> YOLO -> reconstruction"
    )
    parser.add_argument("--mode", choices=["folder", "camera"], default="folder")
    parser.add_argument("--input-dir", type=str, default="sample_data/rpi_demo_inputs")
    parser.add_argument("--output-dir", type=str, default="outputs/rpi_demo")
    parser.add_argument("--camera-index", type=int, default=0)

    parser.add_argument("--esrgan-weights", type=str, required=True)
    parser.add_argument("--yolo-weights", type=str, required=True)

    parser.add_argument("--input-size", type=int, default=416)
    parser.add_argument("--tile-size", type=int, default=416)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-frames", type=int, default=1)

    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running on device: {device}")
    print(f"Mode: {args.mode}")

    if args.mode == "folder":
        process_from_folder(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            esrgan_weights=args.esrgan_weights,
            yolo_weights=args.yolo_weights,
            input_size=args.input_size,
            tile_size=args.tile_size,
            conf=args.conf,
            device=device,
        )
    else:
        process_from_camera(
            camera_index=args.camera_index,
            output_dir=args.output_dir,
            esrgan_weights=args.esrgan_weights,
            yolo_weights=args.yolo_weights,
            input_size=args.input_size,
            tile_size=args.tile_size,
            conf=args.conf,
            device=device,
            max_frames=args.max_frames,
        )


if __name__ == "__main__":
    main()