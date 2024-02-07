from pathlib import Path

CELL_ID_TO_RGB = {0: (0, 0, 0), 1: (254, 244, 51), 2: (43, 205, 253)}
RGB_TO_CELL_ID = {(0, 0, 0): 0, (254, 244, 51): 1, (43, 205, 253): 2}
CELL_ID_TO_NAME = {0: "Background", 1: "BC (Background Cell)", 2: "TC (Tumor Cell)"}
CELL_SGM_CLASSES = ["Background", "Background Cell", "Cancer Cell"]
CELL_ID_TO_SHORT_NAME = {0: "BG", 1: "BC", 2: "TC"}
NUM_CELL_CLASSES = 3
TISSUE_SGM_MPP = 0.8  # 0.8mpp == 12.5x

# Grand Challenge folders were input files can be found
GC_CELL_FPATH = Path("/input/images/cell_patches/")
GC_TISSUE_FPATH = Path("/input/images/tissue_patches/")

GC_METADATA_FPATH = Path("/input/metadata.json")

# Grand Challenge output file
GC_DETECTION_OUTPUT_PATH = Path("/output/cell_classification.json")

# Sample dimensions
SAMPLE_SHAPE = (1024, 1024, 3)
