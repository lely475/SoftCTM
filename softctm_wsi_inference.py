import os
from typing import Literal

from util.wsi_inferer import SoftCTM_WSI_Inferer

if __name__ == "__main__":
    ################ Parameters ###########################################
    mode: Literal["20x", "50x"] = "20x"
    data_path = ""  # Add Data directory containing wsis
    output_path = ""  # Add output path
    tile_size = 1024  # SoftCTM input tile size
    batch_size = 8  # Batch size for inference, lower if RAM is exceeded
    visualize = True  # Creates cell markups
    ######################################################################

    onnx_path = "onnx/cell_detection" + ("_20x.onnx" if mode == "20x" else ".onnx")
    tissue_sgm_onnx_path = "onnx/tissue_sgm.onnx"
    desired_mpp = 0.5 if mode == "20x" else 0.2
    os.makedirs(output_path, exist_ok=True)

    # Infere SoftCTM
    inferer = SoftCTM_WSI_Inferer(onnx_path, tissue_sgm_onnx_path)
    inferer.predict(
        data_path, desired_mpp, tile_size, output_path, batch_size, visualize
    )
