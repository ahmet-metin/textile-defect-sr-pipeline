from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from PIL import Image


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(input_dir: Path) -> list[Path]:
    return sorted(
        [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]
    )


def crop_to_divisible_size(image_rgb, patch_size: int):
    """
    Crop the image from the top-left corner so that both width and height
    become divisible by patch_size.
    """
    h, w = image_rgb.shape[:2]
    size_x = (w // patch_size) * patch_size
    size_y = (h // patch_size) * patch_size

    pil_img = Image.fromarray(image_rgb)
    pil_img = pil_img.crop((0, 0, size_x, size_y))
    return pil_img


def save_patch(output_path: Path, patch_rgb):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    patch_bgr = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), patch_bgr)


def prepare_patches(
    input_dir: str | Path,
    output_dir: str | Path,
    patch_size: int = 416,
    save_cropped_image: bool = False,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    images = list_images(input_dir)
    if not images:
        print(f"No valid images found in: {input_dir}")
        return

    print(f"Found {len(images)} images in {input_dir}")

    total_patches = 0

    for image_path in images:
        print(f"Processing: {image_path.name}")

        image_bgr = cv2.imread(str(image_path), 1)
        if image_bgr is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Crop so both dimensions are divisible by patch_size
        cropped_pil = crop_to_divisible_size(image_rgb, patch_size=patch_size)
        cropped_rgb = cv2.cvtColor(
            cv2.imread(str(image_path), 1), cv2.COLOR_BGR2RGB
        )  # just to initialize variable
        cropped_rgb = None
        cropped_rgb = __import__("numpy").array(cropped_pil)

        if save_cropped_image:
            cropped_dir = output_dir / "cropped_images"
            cropped_dir.mkdir(parents=True, exist_ok=True)
            cropped_path = cropped_dir / f"{image_path.stem}_cropped{image_path.suffix}"
            save_patch(cropped_path, cropped_rgb)

        h, w = cropped_rgb.shape[:2]
        rows = h // patch_size
        cols = w // patch_size

        print(f"  Cropped size: {w}x{h}")
        print(f"  Grid: {rows} x {cols}")

        image_patch_dir = output_dir / image_path.stem
        image_patch_dir.mkdir(parents=True, exist_ok=True)

        patch_count = 0
        for a in range(rows):
            for b in range(cols):
                y0 = a * patch_size
                y1 = y0 + patch_size
                x0 = b * patch_size
                x1 = x0 + patch_size

                single_patch = cropped_rgb[y0:y1, x0:x1]

                patch_name = f"{image_path.stem}_r{a}_c{b}.jpg"
                patch_path = image_patch_dir / patch_name
                save_patch(patch_path, single_patch)

                patch_count += 1
                total_patches += 1

        print(f"  Saved {patch_count} patches to {image_patch_dir}")

    print(f"\nDone. Total patches saved: {total_patches}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare non-overlapping 416x416 patches from input images."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing source images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where patch images will be saved.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=416,
        help="Patch size (default: 416).",
    )
    parser.add_argument(
        "--save-cropped-image",
        action="store_true",
        help="If set, save the cropped full image before patch extraction.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_patches(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        save_cropped_image=args.save_cropped_image,
    )