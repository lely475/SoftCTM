import os
from typing import Tuple

import cv2
import numpy as np

from util.constants import CELL_ID_TO_RGB


def rgb2bgr(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """Convert color in RGB format to BGR"""
    return tuple([rgb[2], rgb[1], rgb[0]])


def visualize_prediction(
    wsi: np.ndarray, pred_cells: np.ndarray, f: float, wsi_name: str, output_path: str
) -> None:
    """
    Marks detected tumor and background cells in the whole slide image,
    saves overlay to file
    """
    if f != 1:
        wsi = cv2.resize(wsi, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    bgr_mask = np.zeros_like(wsi)
    for x, y, label, _ in pred_cells:
        cv2.circle(
            bgr_mask,
            (round(x * f), round(y * f)),
            3,
            rgb2bgr(CELL_ID_TO_RGB[label]),
            -1,
        )

    wsi = cv2.cvtColor(wsi, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(wsi, 1.0, bgr_mask, 0.3, 0)

    # Save segmentation mask and overlay to file
    os.makedirs(f"{output_path}/overlays", exist_ok=True)
    os.makedirs(f"{output_path}/masks", exist_ok=True)
    os.makedirs(f"{output_path}/cell_csvs", exist_ok=True)

    cv2.imwrite(f"{output_path}/overlays/{wsi_name}.jpg", overlay)
    cv2.imwrite(f"{output_path}/masks/{wsi_name}.jpg", bgr_mask)
