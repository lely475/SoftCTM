![OCELOT LOGO](logo/ocelot_banner.png)

# SoftCTM: Cell detection by soft instance segmentation and consideration of cell-tissue interaction

In this repository, you can find the source code to recreate our [Grand Challenge OCELOT 23](https://ocelot2023.grand-challenge.org/) **algorithm container**. 
It provides a multi-organ (trained on kidney, head-and-neck, prostate, stomach, endometrium, and bladder samples) deep-learning model for detecting tumor and background cells in H&E images. This makes it directly applicable for tasks such as tumor content estimation. 

We provide source code to:
1. Run our algorithm on whole slide images.
2. Reproduce our OCELOT 2023 Challenge results.

For more information check out our [paper on the method](https://arxiv.org/abs/2312.12151), as well as [OCELOT Challenge 2023](https://ocelot2023.grand-challenge.org/) and [OCELOT dataset](https://lunit-io.github.io/research/publications/ocelot/).

## Our algorithm
Detecting and classifying cells in histopathology whole-slide images is a core task in computational pathology, as it provides valuable insight into the tumor micro environment. In this work we investigate the impact of ground truth formats on the models performance. Additionally, cell-tissue interactions are considered by providing tissue segmentation predictions as input to the cell detection model. We find that a soft, probability-map instance segmentation ground truth leads to best model performance. Combined with cell-tissue interaction and test-time augmentation we achieve 3rd place on the Overlapped Cell On Tissue (OCELOT) test set with mean F1-Score 0.7172.

![inference_pipeline](https://github.com/lely475/lely475/assets/62755943/74c0b66e-caaf-496c-9ed3-a626aed1da3f)

## Download pretrained models
Please download our pretrained tissue segmentation and cell detection models [here](https://1drv.ms/f/s!Aqry0_PzRNA6gdEhYsuOSooj7PP-Gg?e=9kQiE2).
Create a folder `onnx` on the top-level of the repository and add the models there.

## Whole slide image inference
We provide the script [softctm_wsi_inference.py](softctm_wsi_inference.py) to run WSI inference using our algorithm.
To improve efficiency we are not applying test-time augmentation and do not use a larger FOV for tissue segmentation.
We additionally provide a 20x version of the cell detection algorithm to allow broader applicability and faster throughput.

### Install requirements
Please run to ensure all required packages are installed:
```
pip install -r requirements.txt
```

### Configure parameters
Please adapt the parameters in the script according to your requirements:
```python
mode: Literal["20x", "50x"] = "50x"  # original 50x and retrained 20x version
data_path = ""    # Add data directory containing (only!) wsis
output_path = ""  # Add output path
tile_size = 1024  # SoftCTM input tile size
visualize = True  # Creates cell markups
```

In case you only want to run the algorithm on a region of interest (ROI) instead of the whole slide, you can add logic to load them from file (expected as a 0,1 numpy mask, where 0: ignored region, 1: ROI) in [wsi_inferer.py](https://github.com/lely475/ocelot23algo/blob/f975e726552ead08e48ef04e2cf86eb27422cc47/util/wsi_inferer.py#L229):
```python
roi_mask = load_roi_mask(roi_path="", f=f)  # TODO Add path to your roi mask
```

### Run script
```
python softctm_wsi_inference.py
```

### Script outputs
The script produces the following files:
- `detected_cells.csv`: Dataset-level csv, containing tumor and background cell (tc, bc) counts for each WSI. You can easily compute the tumor purity as: `tpe = tc/(tc+bc)`
- `cell_csvs`: Slide-specific csv files with detailed cell detections for each slide (x-,y-coordinates, class label (1: bc, 2: tc), confidence, re-scale factor). 

If you enabled visualization you the following additional directories exist:
- `overlays`: Original WSI with marked cell detections (tc: blue, bc: yellow).
- `masks`: Mask with detected cells (tc: blue, bc: yellow)

## Reproduce OCELOT results
To reproduce our OCELOT results follow the below steps to create the docker container and infere it on the OCELOT test set.

### Build the docker image

Build our docker image:

```bash
bash build.sh
```

### Testing docker image

The script `test.sh` will create the image, run a container and verify that the output file `cell_classification.json` is generated at the designated directory. To do so, simply run the following command:

```bash
bash test.sh
```

### Export algorithm docker image

Generate the `tar` file containing the algorithm docker image:

```bash
bash export.sh
```

### Run inference

You can run inference using the algorithm docker image with the following command, add your desired input and output paths where indicated:
```bash
docker run --rm \
        --memory="12g" \
        --memory-swap="12g" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        -v your-input-path:/input \
        -v your-output-path:/output/ \
        ocelot23algo
```

The output of the algorithm is a json file that logs all detected cells for each image in the format:
```
{
    "name": "image_id",
    "point": [
        x,
        y,
        class_label
    ],
    "probability": confidence_score
}
```

# Credits
The source code and description is adapted from Lunits [OcelotAlgo23 repository](https://github.com/lunit-io/ocelot23algo).
```
@InProceedings{Ryu_2023_CVPR,
    author = {Ryu, Jeongun and Puche, Aaron Valero and Shin, JaeWoong and Park, Seonwook and Brattoli, Biagio and Lee, Jinhee and Jung, Wonkyung and Cho, Soo Ick and Paeng, Kyunghyun and Ock, Chan-Young and Yoo, Donggeun and Pereira, S\'ergio},
    title = {OCELOT: Overlapped Cell on Tissue Dataset for Histopathology},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2023},
    pages = {23902-23912}
}
```

# Code Usage
If you find this code helpful to your work and want to use it, please cite:
```
@InProceedings{10.1007/978-3-031-55088-1_10,
author="Schoenpflug, Lydia Anette
and Koelzer, Viktor Hendrik",
editor="Ahmadi, Seyed-Ahmad
and Pereira, S{\'e}rgio",
title="SoftCTM: Cell Detection by Soft Instance Segmentation and Consideration of Cell-Tissue Interaction",
booktitle="Graphs in Biomedical Image Analysis, and Overlapped Cell on Tissue Dataset for Histopathology",
year="2024",
publisher="Springer Nature Switzerland",
address="Cham",
pages="109--122",
abstract="Detecting and classifying cells in histopathology H {\&}E stained whole-slide images is a core task in computational pathology, as it provides valuable insight into the tumor microenvironment. In this work we investigate the impact of ground truth formats on the models performance. Additionally, cell-tissue interactions are considered by providing tissue segmentation predictions as input to the cell detection model. We find that a ``soft'', probability-map instance segmentation ground truth leads to best model performance. Combined with cell-tissue interaction and test-time augmentation our Soft Cell-Tissue-Model (SoftCTM) achieves 0.7172 mean F1-Score on the Overlapped Cell On Tissue (OCELOT) test set, achieving the third best overall score in the OCELOT 2023 Challenge. The source code for our approach is made publicly available at https://github.com/lely475/ocelot23algo.",
isbn="978-3-031-55088-1"
}

```
