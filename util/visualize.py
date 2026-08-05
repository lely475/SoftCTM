import os
from typing import Tuple

import cv2
import numpy as np

from util.constants import CELL_ID_TO_RGB
from util.utils import WSI_Info


def rgb2bgr(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Converts an RGB color tuple to BGR."""
    return (rgb[2], rgb[1], rgb[0])


def visualize_prediction(wsi_info: WSI_Info, pred_cells: np.ndarray, output_path: str, level: int) -> None:
    """Marks detected cells on a WSI thumbnail and saves the overlay and mask images to file."""
    wsi = wsi_info.load_wsi(level=level)
    vis_w, vis_h = wsi_info.level_dims[level]
    target_w, target_h = wsi_info.shape_target
    fx, fy = vis_w / target_w, vis_h / target_h

    bgr_mask = np.zeros_like(wsi)
    for x, y, label, _ in pred_cells:
        cv2.circle(
            bgr_mask,
            (round(x * fx), round(y * fy)),
            3,
            rgb2bgr(CELL_ID_TO_RGB[label]),
            -1,
        )

    wsi = cv2.cvtColor(wsi, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(wsi, 1.0, bgr_mask, 0.3, 0)

    os.makedirs(f"{output_path}/overlays", exist_ok=True)
    os.makedirs(f"{output_path}/masks", exist_ok=True)
    cv2.imwrite(f"{output_path}/overlays/{wsi_info.name}.jpg", overlay)
    cv2.imwrite(f"{output_path}/masks/{wsi_info.name}.jpg", bgr_mask)
