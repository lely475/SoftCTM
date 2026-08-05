import logging
import os
from typing import Generator, List, Optional, Tuple
from xml.etree import ElementTree as ET

import cv2
import numpy as np
import openslide
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


def xml_to_mask(xml_path: str, shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Rasterizes an ASAP-style annotation XML into a binary (0/1) mask."""
    xml = open(xml_path).read()
    root = ET.fromstring(xml)
    inclusions, exclusions = [], []
    for region in root.findall(".//Region"):
        points = [(int(v.attrib["X"]), int(v.attrib["Y"])) for v in region.find(".//Vertices")]
        if region.attrib.get("NegativeROA") == "1":
            exclusions.append(points)
        else:
            inclusions.append(points)
    if shape is None:
        xs, ys = zip(*[pt for reg in inclusions + exclusions for pt in reg])
        shape = (max(ys) + 1, max(xs) + 1)
    mask = Image.new("1", shape, 0)
    draw = ImageDraw.Draw(mask)
    for poly in inclusions:
        draw.polygon(poly, outline=1, fill=1)
    for poly in exclusions:
        draw.polygon(poly, outline=0, fill=0)
    return np.array(mask, dtype=np.uint8)


class WSI_Info:
    """Wraps an OpenSlide slide, auto-selecting the pyramid level closest to desired_mpp for tiling and I/O."""

    def __init__(self, wsi_path: str, desired_mpp: float, level: Optional[int] = None) -> None:
        self.path = wsi_path
        self.name = os.path.basename(wsi_path).split(".")[0]
        self.desired_mpp = desired_mpp
        self.slide = openslide.open_slide(self.path)
        self.level_dims = self.slide.level_dimensions
        base_mpp = float(self.slide.properties[openslide.PROPERTY_NAME_MPP_X])
        self.level = self._closest_level(base_mpp, desired_mpp) if level is None else level
        orig_res = base_mpp * self.slide.level_downsamples[self.level]
        self.scale = orig_res / self.desired_mpp

    def _closest_level(self, base_mpp: float, desired_mpp: float) -> int:
        """Picks the most-downsampled pyramid level whose native resolution is still at least as fine as desired_mpp."""
        level_mpps = [base_mpp * d for d in self.slide.level_downsamples]
        fine_enough = [i for i, mpp in enumerate(level_mpps) if mpp <= desired_mpp]
        return max(fine_enough) if fine_enough else 0

    @property
    def level_downsamples(self) -> Tuple[float, ...]:
        return self.slide.level_downsamples

    @property
    def shape_orig(self) -> Tuple[int, int]:
        """Slide dimensions (width, height) at self.level."""
        return self.level_dims[self.level]

    @property
    def shape_target(self) -> Tuple[int, int]:
        """Slide dimensions (width, height) at desired_mpp, independent of self.level."""
        return tuple(round(v * self.scale) for v in self.shape_orig)

    @property
    def mpp(self) -> float:
        """Native (level 0) microns-per-pixel of the slide."""
        return float(self.slide.properties[openslide.PROPERTY_NAME_MPP_X])

    def tile_image(
        self, tile_size: int, overlap: int, roi_mask: Optional[np.ndarray] = None
    ) -> Tuple[List[int], List[int]]:
        """Returns (x, y) top-left coordinates tiling the slide, filtered to tiles overlapping roi_mask if given."""
        w, h = self.shape_orig
        if self.scale != 1:
            tile_size = round(tile_size / self.scale)
            overlap = round(overlap / self.scale)
        stride = tile_size - 2 * overlap
        x_tiles = int(np.ceil(w / stride))
        y_tiles = int(np.ceil(h / stride))

        if roi_mask is not None:
            mask_h, mask_w = roi_mask.shape
            scale_x, scale_y = mask_w / w, mask_h / h

        x_coords, y_coords = [], []
        for y in range(y_tiles):
            for x in range(x_tiles):
                left, right = x * stride, x * stride + tile_size
                top, bottom = y * stride, y * stride + tile_size
                if bottom > h:
                    top, bottom = h - tile_size, h
                if right > w:
                    left, right = w - tile_size, w
                if roi_mask is None:
                    x_coords.append(left)
                    y_coords.append(top)
                    continue
                m_left, m_top = round(left * scale_x), round(top * scale_y)
                m_right = max(m_left + 1, round(right * scale_x))
                m_bottom = max(m_top + 1, round(bottom * scale_y))
                if roi_mask[m_top:m_bottom, m_left:m_right].sum() > 0:
                    x_coords.append(left)
                    y_coords.append(top)
        return x_coords, y_coords

    def load_tile(self, x: int, y: int, tile_size: int, margin: int = 10) -> np.ndarray:
        """Reads one tile_size x tile_size RGB tile at desired_mpp, using a margin to avoid border resize artifacts."""
        width, height = self.shape_orig
        tile_size_orig = round(tile_size / self.scale)
        x_start, y_start = max(0, x - margin), max(0, y - margin)
        x_lower_m, y_lower_m = x - x_start, y - y_start
        x_end = min(width, x + tile_size_orig + margin)
        y_end = min(height, y + tile_size_orig + margin)
        x_upper_m, y_upper_m = x_end - x - tile_size_orig, y_end - y - tile_size_orig
        w, h = x_end - x_start, y_end - y_start

        downsample = self.slide.level_downsamples[self.level]
        tile = self.slide.read_region(
            (round(x_start * downsample), round(y_start * downsample)), self.level, (w, h)
        )
        tile = np.array(tile.convert("RGB"))
        tile = cv2.resize(
            tile,
            dsize=(
                tile_size + round((x_lower_m + x_upper_m) * self.scale),
                tile_size + round((y_lower_m + y_upper_m) * self.scale),
            ),
            interpolation=cv2.INTER_AREA if self.scale < 1 else cv2.INTER_CUBIC,
        )
        tile = tile[
            round(y_lower_m * self.scale) : round(y_lower_m * self.scale) + tile_size,
            round(x_lower_m * self.scale) : round(x_lower_m * self.scale) + tile_size,
        ]
        assert tile.shape == (tile_size, tile_size, 3), (
            f"x_start {x_start}, x_end {x_end}, y_start {y_start}, y_end {y_end}, tile shape {tile.shape}"
        )
        return tile

    def load_tiles(
        self, x_coords: List[int], y_coords: List[int], tile_size: int, margin: int = 10
    ) -> List[np.ndarray]:
        """Reads a batch of tiles, see load_tile."""
        return [self.load_tile(x, y, tile_size, margin) for x, y in zip(x_coords, y_coords)]

    def load_wsi(self, level: int) -> np.ndarray:
        """Reads the full slide as an RGB array at the given pyramid level."""
        level = min(level, self.slide.level_count - 1)
        wsi = self.slide.read_region((0, 0), level, self.slide.level_dimensions[level])
        return np.array(wsi.convert("RGB"))

    def get_thumbnail(self, shape: Tuple[int, int] = (1000, 1000)) -> Image.Image:
        """Returns a thumbnail no larger than shape, preserving aspect ratio."""
        return self.slide.get_thumbnail(shape)


def load_roi_mask(wsi_info: WSI_Info, roi_path: str = "") -> np.ndarray:
    """Loads a binary ROI mask from a png/npy/annotations file, or falls back to automatic tissue detection."""
    if not os.path.exists(roi_path):
        logger.info(f"{wsi_info.name}: no ROI mask file - using automatic Otsu tissue/background detection")
        return detect_tissue_mask(wsi_info)

    ext = os.path.basename(roi_path).split(".")[-1]
    if ext == "png":
        mask = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
        return (mask > 0).astype(np.uint8)
    if ext == "npy":
        mask = np.load(roi_path)
        return (mask > 0).astype(np.uint8)
    if ext == "annotations":
        return xml_to_mask(roi_path, shape=wsi_info.level_dims[0])
    raise ValueError(f"Unsupported ROI mask file format: {roi_path}")


def detect_tissue_mask(wsi_info: WSI_Info, thumbnail_size: int = 2048) -> np.ndarray:
    """Segments tissue from background via Otsu thresholding on a thumbnail, kept at thumbnail resolution."""
    thumb = np.array(wsi_info.get_thumbnail((thumbnail_size, thumbnail_size)).convert("RGB"))
    gray = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, tissue_mask = cv2.threshold(blurred, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return tissue_mask.astype(np.uint8)


def get_vis_level(level_dimensions: List[Tuple[int, int]], max_px_size: int) -> int:
    """Returns the highest-resolution pyramid level whose dimensions both fit within max_px_size."""
    for i, (width, height) in enumerate(level_dimensions):
        if width <= max_px_size and height <= max_px_size:
            return i
    raise ValueError(f"Smallest level dimension {level_dimensions[-1]} is still > max_px_size {max_px_size}")


def data_generator(x_coords: List[int], y_coords: List[int], batch_size: int) -> Generator:
    """Yields (x_coords, y_coords) in chunks of batch_size."""
    for i in range(0, len(x_coords), batch_size):
        x_batch = np.stack(x_coords[i : i + batch_size], axis=0).astype("int").tolist()
        y_batch = np.stack(y_coords[i : i + batch_size], axis=0).astype("int").tolist()
        yield x_batch, y_batch


def softmax(x: np.ndarray, dim: Optional[int] = None) -> np.ndarray:
    """Numerically stable softmax of x along dim."""
    e_x = np.exp(x - np.max(x, axis=dim, keepdims=True))
    return e_x / np.sum(e_x, axis=dim, keepdims=True)


def save_mask_with_tiles(
    wsi_info: WSI_Info,
    roi_mask: Optional[np.ndarray],
    x_coords: List[int],
    y_coords: List[int],
    tile_size: int,
    output_path: str,
    wsi_name: str,
) -> None:
    """Saves a thumbnail with the ROI mask and tile grid overlaid, for visual sanity-checking of tiling/ROI setup."""
    if roi_mask is None:
        return
    tn = wsi_info.get_thumbnail()
    mask_img = np.array(Image.fromarray((roi_mask * 255).astype(np.uint8)).resize(tn.size)) > 0
    gray = np.mean(np.array(tn), axis=2, keepdims=True).astype(np.uint8)
    blended = Image.fromarray(np.where(mask_img[..., None], np.array(tn), gray))
    scale_x = tn.size[0] / wsi_info.shape_orig[0]
    scale_y = tn.size[1] / wsi_info.shape_orig[1]

    draw = ImageDraw.Draw(blended)
    for x, y in zip(x_coords, y_coords):
        rect = [x * scale_x, y * scale_y, (x + tile_size) * scale_x, (y + tile_size) * scale_y]
        draw.rectangle(rect, outline="red", width=1)

    os.makedirs(f"{output_path}/mask_tns", exist_ok=True)
    blended.save(f"{output_path}/mask_tns/{wsi_name}.png")
