from typing import List, Tuple

import cv2
import numpy as np
from skimage import feature


def find_cells(heatmap: np.ndarray, min_distance: int = 3) -> List[Tuple[int, int, int, float]]:
    """Detects cell centers in a [3, H, W] softmax heatmap via local-maxima peak finding, returning (x, y, class, score)."""
    # Adapted from https://github.com/lunit-io/ocelot23algo/blob/main/user/unet_example/unet.py
    bg, pred_wo_bg = heatmap[0], heatmap[1:]
    fg = cv2.GaussianBlur(1.0 - bg, (0, 0), sigmaX=3)
    peaks = feature.peak_local_max(fg, min_distance=min_distance, exclude_border=0, threshold_abs=0.0)

    maxval = np.max(pred_wo_bg, axis=0)
    maxcls_0 = np.argmax(pred_wo_bg, axis=0)

    # Filter out peaks if background score dominates
    peaks = np.array(
        [peak for peak in peaks if bg[peak[0], peak[1]] < maxval[peak[0], peak[1]]]
    )
    if len(peaks) == 0:
        return []

    # Get score and class of the peaks
    scores = maxval[peaks[:, 0], peaks[:, 1]]
    peak_class = maxcls_0[peaks[:, 0], peaks[:, 1]]
    predicted_cells = [
        (x, y, c + 1, float(s)) for [y, x], c, s in zip(peaks, peak_class, scores)
    ]
    return predicted_cells
