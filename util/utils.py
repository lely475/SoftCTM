import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import openslide


def load_roi_mask(level: int, roi_path: str = "") -> Optional[np.ndarray]:
    """
    Load region of interested mask:
    FIXME Adapt code to match your roi mask format
    Mask==1: Area considered for cell detection
    Mask==0: Area ingored for cell detection
    """
    if os.path.exists(roi_path):
        roi_mask = np.load(roi_path)
        assert np.all(
            np.isin(roi_mask, [0, 1])
        ), "ROI Mask should only contain 0 and 1!"
        f = 1 / (2**level)
        roi_mask = cv2.resize(
            roi_mask, dsize=None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST
        )
    else:
        Warning(
            "No region of interest mask found at",
            roi_path,
            "running cell detection on full WSI.",
        )
        roi_mask = None
    return roi_mask


def load_wsi(
    wsi_path: str, desired_mpp: float, level: int = 0
) -> Tuple[np.ndarray, float]:
    """
    Load whole slide image, resize to desired resolution
    Returns wsi and downsample factor
    """

    slide = openslide.open_slide(wsi_path)
    wsi = slide.read_region((0, 0), level, slide.level_dimensions[level])
    wsi = np.array(wsi.convert("RGB"))

    orig_res = float(slide.properties[openslide.PROPERTY_NAME_MPP_X]) * 2**level
    f = orig_res / desired_mpp
    wsi = cv2.resize(
        wsi,
        dsize=None,
        fx=f,
        fy=f,
        interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC,
    )
    return wsi, f


class WSI_Info:
    def __init__(self, wsi_path: str, desired_mpp: float, level: int = 0) -> None:
        self.path = wsi_path
        self.name = os.path.basename(wsi_path).split(".")[0]
        self.desired_mpp = desired_mpp
        self.level = level
        slide = openslide.open_slide(self.path)
        self.level_dims = slide.level_dimensions
        orig_res = (
            float(slide.properties[openslide.PROPERTY_NAME_MPP_X]) * 2**self.level
        )
        self.f = orig_res / self.desired_mpp
        slide.close()

    @property
    def shape_orig(self):
        return self.level_dims[self.level]

    @property
    def shape_target(self):
        shape = self.level_dims[self.level]
        return tuple(round(v * self.f) for v in shape)

    def tile_image(
        self,
        tile_size: int,
        overlap: int,
        roi_mask: Optional[np.ndarray] = None,
    ) -> Tuple[List[np.ndarray], List[int], List[int], np.ndarray]:
        """Tile image into tiles of size tile_size and with specified overlap"""
        w, h = self.shape_orig
        if self.f != 1:
            tile_size = round(tile_size / self.f)
            overlap = round(overlap / self.f)
        stride = tile_size - 2 * overlap
        x_tiles = int(np.ceil(w / stride))
        y_tiles = int(np.ceil(h / stride))

        x_coords = []
        y_coords = []

        for y in range(y_tiles):
            for x in range(x_tiles):
                left = x * stride
                right = left + tile_size
                top = y * stride
                bottom = top + tile_size
                if bottom > h or right > w:
                    if bottom > h:
                        top = h - tile_size
                        bottom = h
                    if right > w:
                        left = w - tile_size
                        right = w

                if roi_mask is None or roi_mask[top:bottom, left:right].sum() > 0:
                    x_coords.append(left)
                    y_coords.append(top)
        return x_coords, y_coords

    def load_tiles(
        self,
        x_coords: List[int],
        y_coords: List[int],
        tile_size: int,
        margin: float = 0.3,
    ) -> List[np.ndarray]:
        """Load tiles from wsi, use overlap to avoid resizing border artifacts"""
        slide = openslide.open_slide(self.path)
        width, height = self.shape_orig
        tiles = []
        margin_px_orig = round(tile_size * margin / self.f)
        tile_size_orig = round(tile_size / self.f)
        for x, y in zip(x_coords, y_coords):
            x_start = max(0, x - margin_px_orig)
            y_start = max(0, y - margin_px_orig)
            x_lower_m, y_lower_m = x - x_start, y - y_start
            x_end = min(width, x + tile_size_orig + margin_px_orig)
            y_end = min(height, y + tile_size_orig + margin_px_orig)
            x_upper_m = x_end - x - tile_size_orig
            y_upper_m = y_end - y - tile_size_orig
            w, h = x_end - x_start, y_end - y_start
            tile = slide.read_region((x_start, y_start), self.level, (w, h))
            tile = np.array(tile.convert("RGB"))
            tile = cv2.resize(
                tile,
                dsize=(
                    tile_size + round((x_lower_m + x_upper_m) * self.f),
                    tile_size + round((y_lower_m + y_upper_m) * self.f),
                ),
                interpolation=cv2.INTER_AREA if self.f < 1 else cv2.INTER_CUBIC,
            )
            tile = tile[
                round(y_lower_m * self.f) : round(y_lower_m * self.f) + tile_size,
                round(x_lower_m * self.f) : round(x_lower_m * self.f) + tile_size,
            ]
            assert tile.shape == (
                tile_size,
                tile_size,
                3,
            ), f"x_start {x_start}, x_end {x_end}, y_start {y_start}, y_end {y_end}, Tile shape {tile.shape}"
            tiles.append(tile)
        slide.close()
        return tiles

    def load_wsi(self, level: int):
        slide = openslide.open_slide(self.path)
        level = slide.level_count - 1 if level > slide.level_count else level
        wsi = slide.read_region((0, 0), level, slide.level_dimensions[level])
        wsi = np.array(wsi.convert("RGB"))
        slide.close()
        return wsi


def get_vis_level(level_dimensions: List[Tuple[int, int]], max_px_size: int) -> int:
    for i, (width, height) in enumerate(level_dimensions):
        if width <= max_px_size and height <= max_px_size:
            return i
    return ValueError(
        f"Smallest level dimension {level_dimensions[-1]} "
        f"is still > than defined max px size: {max_px_size}"
    )


def batch(arr_list: List[np.ndarray], batch_size: int) -> List[np.ndarray]:
    """Batch array into batches of size batch_size"""

    batch_size = len(arr_list) if len(arr_list) < batch_size else batch_size
    return [
        np.stack(arr_list[i : i + batch_size], axis=0)
        for i in range(0, len(arr_list), batch_size)
    ]


def softmax(x, dim=None):
    """
    Compute the softmax function for the input array x
    along the specified dimension dim.
    """
    e_x = np.exp(x - np.max(x, axis=dim, keepdims=True))
    # Subtracting max(x) for numerical stability
    return e_x / np.sum(e_x, axis=dim, keepdims=True)
