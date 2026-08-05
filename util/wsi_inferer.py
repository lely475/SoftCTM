import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch  # noqa: F401  (import before onnxruntime so its bundled CUDA/cuDNN libs are found)
from onnxruntime import InferenceSession
from tqdm import tqdm

from util.constants import NUM_CELL_CLASSES, TISSUE_SGM_MPP
from util.find_cells import find_cells
from util.utils import WSI_Info, data_generator, get_vis_level, load_roi_mask, save_mask_with_tiles, softmax
from util.visualize import visualize_prediction

logger = logging.getLogger(__name__)


def _find_cells_core_tile(
    mask: np.ndarray, args: Tuple[int, int, int, int, int, int, float, int, int]
) -> List[list]:
    """Find cells in a tile of the recombined mask, returning (x, y, class, score) for each detected cell."""
    left, top, x0, y0, x1, y1, min_distance, core_right, core_bottom = args
    window = mask[:, y0:y1, x0:x1]
    pred_cells_tile = np.array(
        find_cells(window.astype("float32"), min_distance=min_distance)
    )
    if np.size(pred_cells_tile) == 0:
        return []
    pred_cells_tile[:, 0] += x0
    pred_cells_tile[:, 1] += y0
    in_core = (
        (pred_cells_tile[:, 0] >= left) & (pred_cells_tile[:, 0] < core_right)
        & (pred_cells_tile[:, 1] >= top) & (pred_cells_tile[:, 1] < core_bottom)
    )
    return pred_cells_tile[in_core].tolist()


class SoftCTM_WSI_Inferer:
    """Runs SoftCTM whole-slide cell detection (Schoenpflug & Koelzer, 2023 - arXiv:2312.12151)."""

    def __init__(self, onnx_path: str, tissue_sgm_onnx_path: Optional[str] = None) -> None:
        """Loads the ONNX cell-detection model, and optionally a tissue-segmentation model, onto CUDA."""
        self._ort_session = InferenceSession(onnx_path, providers=["CUDAExecutionProvider"])
        self._use_tissue_pred = tissue_sgm_onnx_path is not None
        if self._use_tissue_pred:
            self._tissue_sgm_ort_session = InferenceSession(
                tissue_sgm_onnx_path, providers=["CUDAExecutionProvider"]
            )

    def predict_tissue_sgm(self, tiles: List[np.ndarray], desired_mpp: float) -> np.ndarray:
        """Predicts tissue segmentation for a batch of tiles, rescaling to TISSUE_SGM_MPP and back to desired_mpp."""
        scale = desired_mpp / TISSUE_SGM_MPP  # 20x: 0.5mpp -> 12.5x: 0.8mpp
        tiles = [cv2.resize(t, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for t in tiles]
        tiles = (np.moveaxis(np.array(tiles), 3, 1) / 255.0).astype("float32")  # (B,H,W,C) -> (B,C,H,W), normalize

        ort_inputs = {self._tissue_sgm_ort_session.get_inputs()[0].name: tiles}
        pred = softmax(self._tissue_sgm_ort_session.run(None, ort_inputs)[0], dim=1)

        pred_upsampled = [
            cv2.resize(p, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_LINEAR)
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
        """Tiles a WSI, runs SoftCTM per tile, recombines into a slide-level heatmap, and finds cell peaks in it."""

        if not save_predictions or not os.path.exists(
            f"{output_path}/npy/{wsi_info.name}.npy"
        ):
            t0 = time.time()
            pred_cells = []
            width, height = wsi_info.shape_target
            overlap = tile_size // 4

            #  1. Tiling image into tiles of size tile_size x tile_size
            all_x, all_y = wsi_info.tile_image(
                tile_size,
                overlap=overlap,
                roi_mask=roi_mask,
            )
            vis_tile_size = round(tile_size / wsi_info.scale)
            save_mask_with_tiles(wsi_info, roi_mask, all_x, all_y, vis_tile_size, output_path, wsi_info.name)

            # 2. Batch tiles
            assert len(all_x) == len(all_y), f"Mismatch in x_coords and y_coords length: {len(all_x)}!={len(all_y)}"
            num_samples = len(all_x)
            if num_samples == 0:
                logger.warning(f"{wsi_info.name}: no tiles overlap the ROI mask - skipping (0 cells)")
                os.makedirs(f"{output_path}/npy", exist_ok=True)
                pred_cells = np.zeros((0, 4))
                np.save(f"{output_path}/npy/{wsi_info.name}.npy", pred_cells)
                return pred_cells
            bs = num_samples if num_samples < bs else bs

            # From global to local coordinates within the ROI bounding box
            target_x = np.minimum(
                np.round(np.array(all_x) * wsi_info.scale).astype(np.int64), width - tile_size - 1
            )
            target_y = np.minimum(
                np.round(np.array(all_y) * wsi_info.scale).astype(np.int64), height - tile_size - 1
            )
            bbox_x0, bbox_y0 = int(target_x.min()), int(target_y.min())
            bbox_w = int(target_x.max()) + tile_size - bbox_x0
            bbox_h = int(target_y.max()) + tile_size - bbox_y0

            mask = np.zeros((NUM_CELL_CLASSES, bbox_h, bbox_w), dtype=np.float16)
            counter = np.zeros((1, bbox_h, bbox_w), dtype=np.float16)
            mask_gb = (mask.nbytes + counter.nbytes) / 1024**3
            logger.info(
                f"{wsi_info.name}: {num_samples} tiles, "
                f"ROI bbox {bbox_w}x{bbox_h} ({100 * bbox_w * bbox_h / (width * height):.1f}% "
                f"of {width}x{height} full extent), mask+counter ~{mask_gb:.2f} GB"
            )

            # 3. Predict on tiles
            # Prefetch next batch of tiles while predicting on the current batch, to avoid waiting for disk I/O
            batches = list(data_generator(all_x, all_y, bs))
            with ThreadPoolExecutor(max_workers=1) as loader:
                next_tiles = loader.submit(
                    wsi_info.load_tiles, batches[0][0], batches[0][1], tile_size
                )
                for i, (x_coords, y_coords) in enumerate(
                    tqdm(batches, desc="Predicting tiles")
                ):
                    tiles = next_tiles.result()
                    if i + 1 < len(batches):
                        nx, ny = batches[i + 1]
                        next_tiles = loader.submit(
                            wsi_info.load_tiles, nx, ny, tile_size
                        )

                    # Tissue segmentation prediction
                    if self._use_tissue_pred:
                        tissue_pred = self.predict_tissue_sgm(tiles, desired_mpp)

                    # Prepare input format (B, C, H, W)
                    tiles = (np.moveaxis(tiles, 3, 1) / 255.0).astype("float32")
                    if self._use_tissue_pred:
                        tiles = np.concatenate((tiles, tissue_pred), axis=1)

                    # Cell detection prediction
                    ort_inputs = {self._ort_session.get_inputs()[0].name: tiles}
                    logits_sgm = self._ort_session.run(None, ort_inputs)[0]
                    pred_sgm = softmax(logits_sgm, dim=1)

                    # Save tile in the ROI-bounding-box-sized prediction mask
                    for pred, x, y in zip(pred_sgm, x_coords, y_coords):
                        tx, ty = round(x * wsi_info.scale), round(y * wsi_info.scale)
                        assert tx - (width - tile_size - 1) <= 1 and ty - (height - tile_size - 1) <= 1
                        x = min(tx, width - tile_size - 1) - bbox_x0
                        y = min(ty, height - tile_size - 1) - bbox_y0
                        mask[:, y : y + tile_size, x : x + tile_size] += pred
                        counter[:, y : y + tile_size, x : x + tile_size] += 1

            # 4. Recombine tile prediction into mask for the whole slide
            # Average overlapping predictions in place
            np.divide(mask, counter, out=mask, where=counter > 0)
            del counter

            # Set non-ROI area prediction to background
            if roi_mask is not None:
                rh, rw = roi_mask.shape
                rx0, ry0 = int(bbox_x0 * rw / width), int(bbox_y0 * rh / height)
                rx1 = max(rx0 + 1, int(np.ceil((bbox_x0 + bbox_w) * rw / width)))
                ry1 = max(ry0 + 1, int(np.ceil((bbox_y0 + bbox_h) * rh / height)))
                roi_crop = cv2.resize(
                    roi_mask[ry0:ry1, rx0:rx1],
                    dsize=(bbox_w, bbox_h),
                    interpolation=cv2.INTER_NEAREST,
                )
                mask[0, roi_crop == 0] = 1
                mask[1:, roi_crop == 0] = 0

            # Detect cells as peaks in the prediction mask, parallelized for speed
            core = tile_size - 2 * overlap
            min_distance = round(3 / desired_mpp)
            tasks = []
            for top in range(0, bbox_h, core):
                for left in range(0, bbox_w, core):
                    y0, y1 = max(0, top - overlap), min(bbox_h, top + core + overlap)
                    x0, x1 = max(0, left - overlap), min(bbox_w, left + core + overlap)
                    core_right, core_bottom = min(left + core, bbox_w), min(top + core, bbox_h)
                    tasks.append(
                        (left, top, x0, y0, x1, y1, min_distance, core_right, core_bottom)
                    )
            n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                for cells in tqdm(
                    pool.map(lambda t: _find_cells_core_tile(mask, t), tasks),
                    desc="Find cells",
                    total=len(tasks),
                ):
                    pred_cells.extend(cells)

            # Save detected cells to npy, translating from local back to global coordinates
            os.makedirs(f"{output_path}/npy", exist_ok=True)
            pred_cells = np.array(pred_cells)
            if pred_cells.size > 0:
                pred_cells[:, 0] += bbox_x0
                pred_cells[:, 1] += bbox_y0
            np.save(f"{output_path}/npy/{wsi_info.name}.npy", pred_cells)
            logger.info(f"{wsi_info.name}: {len(pred_cells)} cells found in {time.time() - t0:.1f}s")
        else:
            pred_cells = np.load(f"{output_path}/npy/{wsi_info.name}.npy")
            logger.info(f"{wsi_info.name}: loaded {len(pred_cells)} cached cells from npy")
        return pred_cells

    def continue_run(
        self, wsis: List[str], wsi_paths: List[str], output_path: str
    ) -> Tuple[List[str], List[str]]:
        """Drops WSIs already present in output_path/detected_cells.csv, so a rerun only processes the remainder."""
        self._df = pd.read_csv(f"{output_path}/detected_cells.csv")
        done_wsis = self._df["wsi"].tolist()
        missing_wsis = sorted(set(wsis).difference(done_wsis))
        missing_idxs = [wsis.index(wsi) for wsi in missing_wsis]
        wsi_paths = [wsi_paths[i] for i in missing_idxs]
        logger.info(f"Resuming previous run: {len(done_wsis)} WSI(s) already done, {len(missing_wsis)} remaining")
        return missing_wsis, wsi_paths

    def wsi_level_csv(self, pred_cells: np.ndarray, mpp: float, output_path: str) -> None:
        """Writes one CSV of all detected cells (x, y, label, confidence, mpp) for a single WSI."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df = pd.DataFrame(
            {
                "x": pred_cells[:, 0].astype("int"),
                "y": pred_cells[:, 1].astype("int"),
                "label": pred_cells[:, 2].astype("int"),
                "confidence": pred_cells[:, 3],
                "mpp": mpp,
            }
        )
        df.to_csv(output_path, index=False)

    def dataset_level_csv(
        self, wsis: List[str], tc: List[int], bc: List[int], output_path: str
    ) -> None:
        """Appends tumor/background cell counts for wsis to the dataset-level CSV at output_path."""
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
        """Runs cell detection on every WSI in data_path and writes per-WSI and dataset-level result CSVs."""
        wsi_paths = sorted(f for f in glob(f"{data_path}/*") if os.path.isfile(f))
        assert wsi_paths, f"No files were found in {data_path}!"
        wsis = [os.path.basename(f).split(".")[0] for f in wsi_paths]
        logger.info(f"Found {len(wsis)} WSI(s) in {data_path}")
        if os.path.exists(f"{output_path}/detected_cells.csv"):
            wsis, wsi_paths = self.continue_run(wsis, wsi_paths, output_path)

        run_t0 = time.time()
        total_tc = total_bc = 0
        for wsi_file, wsi_name in tqdm(zip(wsi_paths, wsis), desc="Predict wsi", total=len(wsis)):
            wsi_info = WSI_Info(wsi_file, desired_mpp)
            downsample = wsi_info.level_downsamples[wsi_info.level]
            logger.info(
                f"{wsi_name}: native mpp={wsi_info.mpp:.3f}, target mpp={desired_mpp} "
                f"-> reading level {wsi_info.level} (downsample {downsample:.1f}x, "
                f"native shape={wsi_info.level_dims[0]})"
            )
            # TODO Add path to your roi mask; falls back to automatic Otsu tissue detection if left empty.
            roi_mask = load_roi_mask(wsi_info, roi_path="")

            pred_cells = self.predict_wsi(
                wsi_info=wsi_info,
                tile_size=tile_size,
                output_path=output_path,
                bs=batch_size,
                desired_mpp=desired_mpp,
                roi_mask=roi_mask,
                save_predictions=save_predictions,
            )

            tc = np.count_nonzero(pred_cells[:, 2] == 2)
            bc = np.count_nonzero(pred_cells[:, 2] == 1)
            total_tc += tc
            total_bc += bc
            logger.info(f"{wsi_name}: {tc} tumor cells, {bc} background cells")

            if visualize:
                vis_level = get_vis_level(wsi_info.level_dims, max_px_size=30000)
                visualize_prediction(wsi_info, pred_cells, output_path, vis_level)

            self.wsi_level_csv(pred_cells, desired_mpp, f"{output_path}/cell_csvs/{wsi_name}.csv")
            self.dataset_level_csv([wsi_name], [tc], [bc], f"{output_path}/detected_cells.csv")

        logger.info(
            f"Done: {len(wsis)} WSI(s) in {time.time() - run_t0:.1f}s, "
            f"{total_tc} tumor cells / {total_bc} background cells total"
        )
