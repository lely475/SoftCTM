import os
from glob import glob
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from onnxruntime import InferenceSession
from tqdm import tqdm

from util.constants import NUM_CELL_CLASSES, TISSUE_SGM_MPP
from util.find_cells import find_cells
from util.utils import WSI_Info, batch, get_vis_level, load_roi_mask, softmax
from util.visualize import visualize_prediction


class SoftCTM_WSI_Inferer:
    """
        Runs background and tumor cell detection on a whole slide image,
        Using SoftCTM cell detection algorithm:
        @misc{schoenpflug2023softctm,
          title={SoftCTM: Cell detection by soft instance segmentation
          and consideration of cell-tissue interaction},
          author={Lydia A. Schoenpflug and Viktor H. Koelzer},
          year={2023},
          eprint={2312.12151},
          archivePrefix={arXiv},
          primaryClass={eess.IV}
    }
    """

    def __init__(
        self, onnx_path: str, tissue_sgm_onnx_path: Optional[str] = None
    ) -> None:
        """initializes Inferer"""
        self._ort_session = InferenceSession(
            onnx_path, providers=["CUDAExecutionProvider"]
        )
        self._use_tissue_pred = tissue_sgm_onnx_path is not None
        if self._use_tissue_pred:
            self._tissue_sgm_ort_session = InferenceSession(
                tissue_sgm_onnx_path, providers=["CUDAExecutionProvider"]
            )

    def predict_tissue_sgm(
        self, tiles: List[np.ndarray], desired_mpp: float
    ) -> np.ndarray:
        """Generate tissue segmentation prediction for tiles, upscale to desired mpp"""
        # Downscale tiles to tissue sgm mpp
        f = desired_mpp / TISSUE_SGM_MPP  # 20x: 0.5mpp -> 12.5x: 0.8mpp
        tiles = [
            cv2.resize(t, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) for t in tiles
        ]

        # (B, H, W, C) -> (B, C, H, W), input normalization
        tiles = (np.moveaxis(np.array(tiles), 3, 1) / 255.0).astype("float32")
        tissue_sgm_ort_inputs = {self._ort_session.get_inputs()[0].name: tiles}

        # Predict
        logits = self._tissue_sgm_ort_session.run(None, tissue_sgm_ort_inputs)[0]
        pred = softmax(logits, dim=1)

        # Upsample to desired mpp
        pred_upsampled = [
            cv2.resize(p, None, fx=1 / f, fy=1 / f, interpolation=cv2.INTER_LINEAR)
            for p in np.moveaxis(pred, 1, 3)
        ]
        return np.moveaxis(np.array(pred_upsampled), 3, 1)

    def predict_wsi(
        self,
        wsi_info: WSI_Info,
        tile_size: int,
        output_path: str,
        desired_mpp: float,
        bs: int = 8,
        roi_mask: Optional[np.ndarray] = None,
        save_predictions: bool = True,
    ) -> np.ndarray:
        """Detect cells in a whole slide image by:
        1. Tiling image into tiles of size tile_size x tile_size
        2. Batch tiles
        3. Predict on tiles with SoftCTM
        4. Recombine tile prediction into whole slide level prediction"""

        if not save_predictions or not os.path.exists(
            f"{output_path}/npy/{wsi_info.name}.npy"
        ):
            pred_cells = []
            width, height = wsi_info.shape_target
            mask = np.zeros(
                (NUM_CELL_CLASSES, height, width),
                dtype=np.float16,
            )
            counter = np.zeros_like(mask)

            #  1. Tiling image into tiles of size tile_size x tile_size
            all_x, all_y = wsi_info.tile_image(
                tile_size,
                overlap=tile_size // 4,
                roi_mask=roi_mask,
            )

            # 2. Batch tiles
            b_x, b_y = batch(all_x, bs), batch(all_y, bs)

            # 3. Predict on tiles
            for x_coords, y_coords in tqdm(
                zip(b_x, b_y), desc=f"Predict on {wsi_info.name} tiles", total=len(b_x)
            ):
                # Load tiles
                tiles = wsi_info.load_tiles(x_coords, y_coords, tile_size)

                # Tissue segmentation prediction
                if self._use_tissue_pred:
                    tissue_pred = self.predict_tissue_sgm(tiles, desired_mpp)

                # Prepare input format (B, C, H, W)
                tiles = (np.moveaxis(tiles, 3, 1) / 255.0).astype("float32")
                if self._use_tissue_pred:
                    tiles = np.concatenate((tiles, tissue_pred), axis=1)

                # Cell detection prediction
                ort_inputs = {self._ort_session.get_inputs()[0].name: np.array(tiles)}
                logits_sgm = self._ort_session.run(None, ort_inputs)[0]
                pred_sgm = softmax(logits_sgm, dim=1)

                # Save tile in full wsi prediction mask
                for pred, x, y in zip(pred_sgm, x_coords, y_coords):
                    assert round(x * wsi_info.f) - (width - tile_size - 1) <= 1
                    assert round(y * wsi_info.f) - (height - tile_size - 1) <= 1
                    x = min(round(x * wsi_info.f), width - tile_size - 1)
                    y = min(round(y * wsi_info.f), height - tile_size - 1)
                    mask[:, y : y + tile_size, x : x + tile_size] += pred
                    counter[:, y : y + tile_size, x : x + tile_size] += 1

            # 4. Recombine tile prediction into mask for the whole slide
            # Average overlapping predictions
            mask = np.divide(mask, counter)

            # Set non-ROI area prediction to 100% background
            if roi_mask is not None:
                mask[0, roi_mask == 0] = 1
                mask[1:, roi_mask == 0] = 0

            # Detect cells in each tile
            for x_coords, y_coords in tqdm(
                zip(b_x, b_y), desc="Find cells", total=len(b_x)
            ):
                for x, y in zip(x_coords, y_coords):
                    x, y = round(x * wsi_info.f), round(y * wsi_info.f)
                    pred = mask[:, y : y + tile_size, x : x + tile_size]
                    pred_cells_tile = np.array(
                        find_cells(
                            pred.astype("float32"), min_distance=round(3 / desired_mpp)
                        )
                    )
                    if np.size(pred_cells_tile) > 0:
                        pred_cells_tile[:, 0] += x
                        pred_cells_tile[:, 1] += y
                        pred_cells.extend(pred_cells_tile.tolist())

            # Save detected cells to npy
            os.makedirs(f"{output_path}/npy", exist_ok=True)
            pred_cells = np.array(pred_cells)
            np.save(f"{output_path}/npy/{wsi_info.name}.npy", pred_cells)
        else:
            pred_cells = np.load(f"{output_path}/npy/{wsi_info.name}.npy")
        return pred_cells

    def continue_run(
        self, wsis: List[str], wsi_paths: List[str], output_path: str
    ) -> Tuple[List[str], List[str]]:
        """Continues from previous run"""
        print("Continue from previous run...")
        self._df = pd.read_csv(f"{output_path}/detected_cells.csv")
        done_wsis = self._df["wsi"].tolist()
        missing_wsis = sorted(list(set(wsis).difference(done_wsis)))
        missing_idxs = [wsis.index(wsi) for wsi in missing_wsis]
        wsi_paths = [wsi_paths[i] for i in missing_idxs]
        wsis = missing_wsis
        return wsis, wsi_paths

    def wsi_level_csv(self, pred_cells: np.ndarray, f: float, output_path: str) -> None:
        """Generates and saves csv with all cells detected in wsi"""
        df = pd.DataFrame(
            {
                "x": pred_cells[:, 0],
                "y": pred_cells[:, 1],
                "label": pred_cells[:, 2].astype("int"),
                "confidence": pred_cells[:, 3],
                "f": [f] * int(pred_cells.shape[0]),
            }
        )
        df.to_csv(output_path, index=False)

    def dataset_level_csv(
        self, wsis: List[str], tc: List[int], bc: List[int], output_path: str
    ) -> None:
        """Creates csv with number of detected background and tumor cells in each wsi"""
        new_df = pd.DataFrame({"wsi": wsis, "tc": tc, "bc": bc})
        try:
            self._df = pd.concat((self._df, new_df))
        except AttributeError:
            self._df = new_df
        self._df = self._df.sort_values(by="wsi")
        self._df.to_csv(output_path, index=False)

    def predict(
        self,
        data_path: str,
        desired_mpp: float,
        tile_size: int,
        output_path: str,
        batch_size: int = 8,
        visualize: bool = True,
        save_predictions: bool = True,
    ) -> None:
        """
        Iterates through all wsis in data_path and:
        1. Loads WSI (and region of interest (ROI) mask if provided)
        2. Detects cells with SoftCTM
        3. Logs results to file
        """
        tc = []
        bc = []
        wsi_paths = sorted(glob(f"{data_path}/*"))
        assert len(wsi_paths) > 0, f"No files were found in {wsi_paths}!"
        wsis = [os.path.basename(f).split(".")[0] for f in wsi_paths]
        if os.path.exists(f"{output_path}/detected_cells.csv"):
            wsis, wsi_paths = self.continue_run(wsis, wsi_paths, output_path)

        for idx, (wsi_file, wsi_name) in tqdm(
            enumerate(zip(wsi_paths, wsis)), desc="Predict wsi", total=len(wsis)
        ):
            # 1. Load WSI Info
            wsi_info = WSI_Info(wsi_file, desired_mpp)

            roi_mask = load_roi_mask(
                roi_path="", level=wsi_info.level
            )  # TODO Add path to your roi mask
            if roi_mask is not None:
                assert (
                    wsi_info.shape_orig == roi_mask.shape()
                ), "WSI and ROI mask shape do not match!"

            # 2. Detect cells with SoftCTM
            pred_cells = self.predict_wsi(
                wsi_info=wsi_info,
                tile_size=tile_size,
                output_path=output_path,
                bs=batch_size,
                desired_mpp=desired_mpp,
                roi_mask=roi_mask,
                save_predictions=save_predictions,
            )
            tc.append(np.count_nonzero(pred_cells[:, 2] == 2))
            bc.append(np.count_nonzero(pred_cells[:, 2] == 1))

            if visualize:
                vis_level = get_vis_level(wsi_info.level_dims, max_px_size=30000)
                visualize_prediction(wsi_info, pred_cells, output_path, vis_level)

            # 3. Log results to file
            self.wsi_level_csv(
                pred_cells, wsi_info.f, f"{output_path}/cell_csvs/{wsi_name}.csv"
            )
            self.dataset_level_csv(
                wsis[: idx + 1], tc, bc, f"{output_path}/detected_cells.csv"
            )
