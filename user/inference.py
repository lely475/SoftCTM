import numpy as np
from onnxruntime import InferenceSession
import cv2
from skimage import feature
from typing import List, Tuple


def softmax(x, axis=None):
    """Compute the softmax function for the input array x along the specified dimension axis."""
    e_x = np.exp(
        x - np.max(x, axis=axis, keepdims=True)
    )  # Subtracting max(x) for numerical stability
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class Model():
    """
    Parameters
    ----------
    metadata: Dict
        Dataset metadata in case you wish to compute statistics

    """
    def __init__(self, metadata):
        self.metadata = metadata
        self.tissue_onnx = InferenceSession("onnx/tissue_sgm.onnx", providers=["CUDAExecutionProvider"])
        self.cell_onnx = InferenceSession("onnx/points.onnx", providers=["CUDAExecutionProvider"])

    def __call__(self, cell_patch, tissue_patch, pair_id):
        """This function detects the cells in the cell patch. Additionally
        the broader tissue context is provided. 

        NOTE: this implementation offers a dummy inference example. This must be
        updated by the participant.

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
        # Getting the metadata corresponding to the patch pair ID
        meta_pair = self.metadata[pair_id]

        #############################################
        #### YOUR INFERENCE ALGORITHM GOES HERE #####
        #############################################
        
        tissue_patch = np.expand_dims((tissue_patch / 255.0).astype("float32"), axis=0)
        tissue_patch = {self.tissue_onnx.get_inputs()[0].name: tissue_patch}
        tissue_logits = self.tissue_onnx.run(None, tissue_patch)[0]
        tissue_pred = softmax(tissue_logits, axis=1)

        cell_patch = np.expand_dims((cell_patch / 255.0).astype("float32"), axis=0)
        combined_input = np.expand_dims(np.concatenate((cell_patch, tissue_pred), axis=1), axis=0)
        cell_patch = {self.cell_onnx.get_inputs()[0].name: combined_input}
        cell_logits = self.cell_onnx.run(None, cell_patch)[0]
        cell_pred = np.squeeze(softmax(cell_logits, axis=1))
        detected_cells = self.find_cells(cell_pred)

        #############################################
        ####### RETURN RESULS PER SAMPLE ############
        #############################################

        # We need to return a list of tuples with 4 elements, i.e.:
        # - int: cell's x-coordinate in the cell patch
        # - int: cell's y-coordinate in the cell patch
        # - int: class id of the cell, either 1 (BC) or 2 (TC)
        # - float: confidence score of the predicted cell
        return detected_cells


    def find_cells(heatmap: np.ndarray) -> List[Tuple[int, int, int, float]]:
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
        arr = np.transpose(arr, (1, 2, 0)) # CHW -> HWC

        # Background and non-background channels
        bg, pred_wo_bg = np.split(arr, (1,), axis=0)
        bg = np.squeeze(bg, axis=0)
        obj = 1.0 - bg

        arr = cv2.GaussianBlur(obj, (0, 0), sigmaX=3)
        arr = arr.squeeze().cpu().detach().numpy()

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

        predicted_cells = [(x, y, c + 1, float(s)) for [y, x], c, s in zip(peaks, peak_class, scores)]
        return predicted_cells
    

    def find_cells_full_mask(heatmap: np.ndarray) -> List[Tuple[int, int, int, float]]:
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
        arr = np.transpose(arr, (1, 2, 0)) # CHW -> HWC

        # Background and non-background channels
        bg, pred_wo_bg = np.split(arr, (1,), axis=0)
        bg = np.squeeze(bg, axis=0)
        obj = 1.0 - bg

        arr = cv2.GaussianBlur(obj, (0, 0), sigmaX=3)
        arr = arr.squeeze().cpu().detach().numpy()

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

        predicted_cells = [(x, y, c + 1, float(s)) for [y, x], c, s in zip(peaks, peak_class, scores)]
        return predicted_cells