from calendar import c
from math import e
import torch
import onnxruntime as ort
import numpy as np
import cv2
from skimage import feature
from typing import Any, List, Tuple, Dict


def softmax(x, axis=None):
    """Compute the softmax function for the input array x along the specified dimension axis."""
    e_x = np.exp(
        x - np.max(x, axis=axis, keepdims=True)
    )  # Subtracting max(x) for numerical stability
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class Model:
    """
    Parameters
    ----------
    metadata: Dict
        Dataset metadata in case you wish to compute statistics

    """

    def __init__(self, metadata: Dict[str, Any]):
        self.metadata = metadata
        self.cell_onnx = ort.InferenceSession(
            "onnx/cell_detection.onnx",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.tissue_onnx = ort.InferenceSession(
            "onnx/tissue_sgm.onnx",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    def __call__(self, cell_patch: np.ndarray, tissue_patch: np.ndarray, pair_id: str):
        """This function detects the cells in the cell patch. Additionally
        the broader tissue context is provided.

        Parameters
        ----------
        cell_patch: np.ndarray[uint8]
            Cell patch with shape [1024, 1024, 3] with values from 0 - 255
        tissue_patch: np.ndarray[uint8]
            Tissue patch with shape [1024, 1024, 3] with values from 0 - 255
        pair_id: str
            Identification number of the patch pair

        Returns
        -------
            List[tuple]: for each predicted cell we provide the tuple (x, y, cls, score)
        """

        meta_pair = self.metadata[pair_id]

        # tissue sgm on tissue patch + test time augmentation
        tissue_patch = (tissue_patch / 255.0).astype("float32")
        tissue_patch_tta = self.geometric_test_time_augmentation(tissue_patch)
        tissue_patch_tta = np.moveaxis(tissue_patch_tta, source=-1, destination=1)
        tissue_pred_tta = []

        for tissue_patch in tissue_patch_tta:
            tissue_model_input = {
                self.tissue_onnx.get_inputs()[0].name: np.expand_dims(
                    tissue_patch, axis=0
                )
            }
            tissue_pred_single = self.tissue_onnx.run(None, tissue_model_input)[0]
            tissue_pred_single = softmax(tissue_pred_single, axis=1)
            tissue_pred_tta.append(np.squeeze(tissue_pred_single))

        tissue_pred = np.expand_dims(self.reverse_tta(tissue_pred_tta), axis=0).astype(
            "float32"
        )

        # crop and upsample tissue patch to cell patch size and resolution
        tissue_pred = self.crop_tissue_sample_for_cell_sgm(tissue_pred, meta_pair)
        tissue_pred = np.moveaxis(tissue_pred, source=1, destination=-1)
        tissue_pred_tta = self.geometric_test_time_augmentation(np.squeeze(tissue_pred))

        # prepare cell patch, combine with tissue pred
        cell_patch = (cell_patch / 255.0).astype("float32")
        cell_patch_tta = self.geometric_test_time_augmentation(cell_patch)
        cell_model_input = np.concatenate((cell_patch_tta, tissue_pred_tta), axis=-1)
        cell_model_input = np.moveaxis(cell_model_input, source=-1, destination=1)

        # cell detection on (cell patch, tissue_pred) + test time augmentation
        cell_preds_tta = []
        for model_input in cell_model_input:
            model_input = {
                self.cell_onnx.get_inputs()[0].name: np.expand_dims(model_input, axis=0)
            }
            cell_pred = self.cell_onnx.run(None, model_input)[0]
            cell_preds_tta.append(np.squeeze(cell_pred))

        cell_pred = self.reverse_tta(cell_preds_tta).astype("float32")
        detected_cells = self.find_cells(cell_pred)
        return detected_cells

    def geometric_test_time_augmentation(self, img: np.ndarray) -> List[np.ndarray]:
        """Return all 8 possible geometric transformations of an image"""

        transformed = []
        for flip in [None, 1]:
            for rotate in [
                None,
                cv2.ROTATE_90_CLOCKWISE,
                cv2.ROTATE_180,
                cv2.ROTATE_90_COUNTERCLOCKWISE,
            ]:
                t_img = cv2.flip(img, flip) if flip is not None else img
                t_img = cv2.rotate(t_img, rotate) if rotate is not None else t_img
                transformed.append(t_img)
        return transformed

    def reverse_tta(self, pred: np.ndarray) -> np.ndarray:
        """Combine test-time augmentation predictions into a single prediction"""
        i = 0
        pred = torch.Tensor(np.array(pred))
        for flip in [None, 2]:
            for rotate in [None, 1, 2, 3]:
                if rotate:
                    pred[i] = torch.rot90(pred[i], k=rotate, dims=(1, 2))
                if flip is not None:
                    pred[i] = torch.flip(pred[i], dims=[flip])
                i += 1
        mean_pred = torch.mean(pred, dim=0)
        return mean_pred.numpy()

    def find_cells(self, heatmap: np.ndarray) -> List[Tuple[int, int, int, float]]:
        """This function detects the cells in the output heatmap
        From: https://github.com/lunit-io/ocelot23algo/blob/main/user/unet_example/unet.py

        Parameters
        ----------
        heatmap: np.ndarray
            output heatmap of the model,  shape: [3, 1024, 1024]

        Returns
        -------
            List[tuple]: for each predicted cell we provide the tuple (x, y, cls, score)
        """
        predicted_cells = []
        arr = heatmap

        # Background and non-background channels
        bg, pred_wo_bg = np.split(arr, (1,), axis=0)
        bg = np.squeeze(bg, axis=0)
        obj = 1.0 - bg
        arr = cv2.GaussianBlur(obj, (0, 0), sigmaX=3)

        # List[y, x]
        peaks = feature.peak_local_max(
            arr, min_distance=3, exclude_border=0, threshold_abs=0.0
        )

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

    def crop_tissue_sample_for_cell_sgm(
        self,
        tissue_pred: np.ndarray,
        meta_pair: Dict[str, Any],
    ) -> np.ndarray:
        """Crop tissue sample to cell patch size and resolution"""
        shape = tissue_pred.shape[2:]
        tissue_pred = np.moveaxis(np.squeeze(tissue_pred), source=0, destination=-1)
        cell_x_start = meta_pair["cell"]["x_start"]
        cell_y_start = meta_pair["cell"]["y_start"]
        tissue_x_start = meta_pair["tissue"]["x_start"]
        tissue_y_start = meta_pair["tissue"]["y_start"]
        tissue_x_end = meta_pair["tissue"]["x_end"]
        tissue_y_end = meta_pair["tissue"]["y_end"]

        x_offset = int(
            shape[0] * (cell_x_start - tissue_x_start) / (tissue_x_end - tissue_x_start)
        )
        y_offset = int(
            shape[0] * (cell_y_start - tissue_y_start) / (tissue_y_end - tissue_y_start)
        )

        tissue_pred_excerpt = tissue_pred[
            y_offset : y_offset + shape[0] // 4,
            x_offset : x_offset + shape[1] // 4,
            :,
        ]
        tissue_pred_excerpt = cv2.resize(
            tissue_pred_excerpt,
            dsize=shape,
            interpolation=cv2.INTER_CUBIC,
        )
        tissue_pred_excerpt = np.expand_dims(
            np.moveaxis(tissue_pred_excerpt, source=-1, destination=0), axis=0
        )
        return tissue_pred_excerpt
