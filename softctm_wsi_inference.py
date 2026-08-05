import logging
import os
from typing import Literal

from util.wsi_inferer import SoftCTM_WSI_Inferer

logger = logging.getLogger(__name__)

# ---- Configure your run here -----------------------------------------------
MODE: Literal["20x", "50x"] = "20x"  # original 50x model, or the retrained 20x model
DATA_PATH = ""    # directory containing (only) WSIs
OUTPUT_PATH = ""  # directory to save cell-detection results (csvs, overlays, masks)
TILE_SIZE = 1024  # SoftCTM input tile size
BATCH_SIZE = 8    # lower if you run out of GPU/CPU memory
VISUALIZE = True  # save cell-detection overlay images
ONNX_DIR = "onnx" # default location, adapt if needed
# -----------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    logging.captureWarnings(True)  # route e.g. numpy RuntimeWarnings through the same log format

    onnx_path = f"{ONNX_DIR}/cell_detection_20x.onnx" if MODE == "20x" else f"{ONNX_DIR}/cell_detection.onnx"
    tissue_sgm_onnx_path = f"{ONNX_DIR}/tissue_sgm.onnx"
    desired_mpp = 0.5 if MODE == "20x" else 0.2
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    logger.info(
        f"SoftCTM WSI inference: mode={MODE}, desired_mpp={desired_mpp}, tile_size={TILE_SIZE}, "
        f"batch_size={BATCH_SIZE}, visualize={VISUALIZE}"
    )
    logger.info(f"data_path={DATA_PATH}")
    logger.info(f"output_path={OUTPUT_PATH}")

    inferer = SoftCTM_WSI_Inferer(onnx_path, tissue_sgm_onnx_path)
    inferer.predict(DATA_PATH, desired_mpp, TILE_SIZE, OUTPUT_PATH, BATCH_SIZE, VISUALIZE)


if __name__ == "__main__":
    main()
