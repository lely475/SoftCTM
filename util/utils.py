import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import openslide


def load_roi_mask(f: float, roi_path: str = "") -> Optional[np.ndarray]:
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


def tile_image(
    image: np.ndarray,
    tile_size: int,
    overlap: int,
    roi_mask: Optional[np.ndarray] = None,
) -> Tuple[List[np.ndarray], List[int], List[int], np.ndarray]:
    """Tile image into tiles of size tile_size and with specified overlap"""

    h, w = image.shape[:2]
    stride = tile_size - 2 * overlap

    x_tiles = int(np.ceil(w / stride))
    y_tiles = int(np.ceil(h / stride))

    tiles = []
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
                tiles.append(image[top:bottom, left:right])
                x_coords.append(left)
                y_coords.append(top)
    return tiles, x_coords, y_coords


def batch(arr_list: List[np.ndarray], batch_size: int) -> List[np.ndarray]:
    """Batch array into batches of size batch_size"""

    batch_size = len(arr_list) if len(arr_list) < batch_size else batch_size
    return [
        np.stack(arr_list[i : i + batch_size], axis=0)
        for i in range(0, len(arr_list), batch_size)
    ]


def softmax(x, dim=None):
    """Compute the softmax function for the input array x along the specified dimension dim."""
    e_x = np.exp(x - np.max(x, axis=dim, keepdims=True))
    # Subtracting max(x) for numerical stability
    return e_x / np.sum(e_x, axis=dim, keepdims=True)
