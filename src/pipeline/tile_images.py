from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np


@dataclass
class TileInfo:
    tile_id: int
    row: int
    col: int
    x0: int
    y0: int
    x1: int
    y1: int
    tile_path: str | None = None


def load_image(image_path: str | Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_rgb_image(image: np.ndarray, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def tile_image_non_overlapping(
    image: np.ndarray,
    tile_size: int = 416,
    expected_grid: Tuple[int, int] | None = (4, 4),
) -> Tuple[List[np.ndarray], List[TileInfo]]:
    """
    Split an image into non-overlapping tiles.

    Paper-aligned behavior:
    - tile_size = 416
    - overlap = 0
    - stride = tile_size
    - for SR output 1664x1664 => 4x4 grid => 16 tiles total

    Parameters
    ----------
    image : np.ndarray
        RGB image, shape (H, W, 3)
    tile_size : int
        Tile size in pixels
    expected_grid : tuple[int, int] | None
        Optional check for expected rows/cols

    Returns
    -------
    tiles : list[np.ndarray]
        List of RGB tiles
    infos : list[TileInfo]
        Tile metadata with coordinates in the full image
    """
    h, w = image.shape[:2]

    if h % tile_size != 0 or w % tile_size != 0:
        raise ValueError(
            f"Image size must be divisible by tile_size with overlap=0. "
            f"Got image size {(w, h)} and tile_size={tile_size}."
        )

    rows = h // tile_size
    cols = w // tile_size

    if expected_grid is not None and (rows, cols) != expected_grid:
        raise ValueError(
            f"Expected grid {expected_grid}, but got {(rows, cols)} "
            f"from image shape {(h, w)} and tile_size={tile_size}."
        )

    tiles: List[np.ndarray] = []
    infos: List[TileInfo] = []

    tile_id = 0
    for row in range(rows):
        for col in range(cols):
            y0 = row * tile_size
            y1 = y0 + tile_size
            x0 = col * tile_size
            x1 = x0 + tile_size

            tile = image[y0:y1, x0:x1].copy()
            tiles.append(tile)
            infos.append(
                TileInfo(
                    tile_id=tile_id,
                    row=row,
                    col=col,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                )
            )
            tile_id += 1

    return tiles, infos


def save_tiles(
    tiles: List[np.ndarray],
    infos: List[TileInfo],
    output_dir: str | Path,
    stem: str,
) -> List[TileInfo]:
    """
    Save tiles with stable names preserving the original sequence.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    updated_infos: List[TileInfo] = []

    for tile, info in zip(tiles, infos):
        tile_name = f"{stem}_tile_{info.tile_id:02d}_r{info.row}_c{info.col}.png"
        tile_path = output_dir / tile_name
        save_rgb_image(tile, tile_path)

        updated_infos.append(
            TileInfo(
                tile_id=info.tile_id,
                row=info.row,
                col=info.col,
                x0=info.x0,
                y0=info.y0,
                x1=info.x1,
                y1=info.y1,
                tile_path=str(tile_path),
            )
        )

    return updated_infos


def reconstruct_from_tiles(
    tiles: List[np.ndarray],
    infos: List[TileInfo],
    full_shape: Tuple[int, int, int],
) -> np.ndarray:
    """
    Reassemble tiles in the same order/positions into the original full-size image.
    """
    canvas = np.zeros(full_shape, dtype=np.uint8)

    for tile, info in zip(tiles, infos):
        canvas[info.y0:info.y1, info.x0:info.x1] = tile

    return canvas


def translate_box_to_image_coords(
    box_xyxy: Tuple[float, float, float, float],
    info: TileInfo,
) -> Tuple[float, float, float, float]:
    """
    Convert a tile-local box into full-image coordinates by adding tile offsets.
    """
    x1, y1, x2, y2 = box_xyxy
    return (
        x1 + info.x0,
        y1 + info.y0,
        x2 + info.x0,
        y2 + info.y0,
    )


def translate_polygon_to_image_coords(
    polygon_xy: List[Tuple[float, float]],
    info: TileInfo,
) -> List[Tuple[float, float]]:
    """
    Convert tile-local polygon/mask boundary points into full-image coordinates.
    """
    return [(x + info.x0, y + info.y0) for x, y in polygon_xy]


def example_usage(
    image_path: str | Path,
    output_dir: str | Path,
    tile_size: int = 416,
) -> Dict[str, Any]:
    """
    Example helper:
    - reads image
    - splits into 4x4 non-overlapping 416x416 tiles
    - saves tiles
    - reconstructs them back to verify correctness
    """
    image = load_image(image_path)
    stem = Path(image_path).stem

    tiles, infos = tile_image_non_overlapping(
        image=image,
        tile_size=tile_size,
        expected_grid=(4, 4),
    )

    infos = save_tiles(tiles, infos, output_dir=output_dir, stem=stem)

    reconstructed = reconstruct_from_tiles(
        tiles=tiles,
        infos=infos,
        full_shape=image.shape,
    )

    recon_path = Path(output_dir) / f"{stem}_reconstructed_check.png"
    save_rgb_image(reconstructed, recon_path)

    return {
        "num_tiles": len(tiles),
        "tile_paths": [i.tile_path for i in infos],
        "reconstruction_check": str(recon_path),
    }


if __name__ == "__main__":
    # Example:
    # python src/pipeline/tile_images.py
    # after editing the paths below.
    demo_image = "sample_data/demo_sr_image.png"
    demo_output = "outputs/tiles_demo"

    result = example_usage(
        image_path=demo_image,
        output_dir=demo_output,
        tile_size=416,
    )
    print(result)