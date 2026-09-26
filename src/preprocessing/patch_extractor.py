"""
Patch extraction engine for digital pathology WSIs.

Extracts patches at specified magnification (default 20x) with:
1. Automatic tissue filtering via TissueDetector.
2. Dual annotation generation: YOLO bounding box txt and binary mask png.
3. Negative patch subsampling to control background ratio.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import concurrent.futures
import cv2
import numpy as np
import openslide
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from src.preprocessing.tissue_detector import TissueDetector
from src.utils.xml_parser import ASAPAnnotation


@dataclass
class ExtractedPatch:
    """Container for an extracted WSI patch and its dual ground truth labels."""
    patch_id: str
    slide_id: str
    x0_l0: int
    y0_l0: int
    size_l0: int
    patch_size: int
    image: np.ndarray                                   # (H, W, 3) RGB uint8
    mask: np.ndarray                                    # (H, W) uint8, values 0 or 255
    yolo_bboxes: List[Tuple[int, float, float, float, float]]  # (class_id, cx, cy, w, h)
    num_glomeruli: int
    tissue_ratio: float
    is_positive: bool


class PatchExtractor:
    """Extracts synchronized image patches, masks, and YOLO labels from a WSI."""

    def __init__(
        self,
        slide: openslide.OpenSlide,
        slide_id: str,
        annotations: List[ASAPAnnotation],
        tissue_detector: TissueDetector,
        target_mag: int = 20,
        base_mag: int = 40,
        patch_size: int = 1024,
        stride: int = 768,
        min_tissue_ratio: float = 0.15,
        min_glom_overlap_ratio: float = 0.20,
        negative_sample_ratio: float = 1.0,
        random_seed: int = 42,
    ) -> None:
        self.slide = slide
        self.slide_id = slide_id
        self.annotations = annotations
        self.tissue_detector = tissue_detector

        self.downsample = base_mag / float(target_mag)
        self.patch_size = patch_size
        self.patch_size_l0 = int(round(patch_size * self.downsample))
        self.stride_l0 = int(round(stride * self.downsample))

        self.min_tissue_ratio = min_tissue_ratio
        self.min_glom_overlap_ratio = min_glom_overlap_ratio
        self.negative_sample_ratio = negative_sample_ratio
        self.rng = np.random.default_rng(random_seed)

        self.w_l0, self.h_l0 = slide.dimensions
        self.strtree = STRtree([a.geometry for a in self.annotations]) if len(self.annotations) > 0 else None

    def plan_patches(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Scan the WSI grid and identify candidate positive and negative patch locations.
        Does not read pixel data yet, ensuring fast planning.
        """
        positive_candidates: List[Dict[str, Any]] = []
        negative_candidates: List[Dict[str, Any]] = []

        # Iterate through the grid at Level 0
        y_max = self.h_l0 - self.patch_size_l0
        x_max = self.w_l0 - self.patch_size_l0

        for y0 in range(0, y_max + 1, self.stride_l0):
            for x0 in range(0, x_max + 1, self.stride_l0):
                # 1. Fast tissue check on thumbnail
                tissue_ratio = self.tissue_detector.get_tissue_ratio(
                    x0, y0, self.patch_size_l0, self.patch_size_l0
                )
                if tissue_ratio < self.min_tissue_ratio:
                    continue

                patch_geom = box(x0, y0, x0 + self.patch_size_l0, y0 + self.patch_size_l0)

                # 2. Check overlap with annotations using spatial index
                intersecting_annots: List[Tuple[ASAPAnnotation, BaseGeometry, float]] = []
                if self.strtree is not None:
                    hit_indices = self.strtree.query(patch_geom, predicate="intersects")
                    for idx in hit_indices:
                        annot = self.annotations[idx]
                        try:
                            inter = annot.geometry.intersection(patch_geom)
                            if not inter.is_empty and annot.area > 0:
                                overlap_ratio = float(inter.area / annot.area)
                                intersecting_annots.append((annot, inter, overlap_ratio))
                        except Exception:
                            continue

                # Qualifying glomeruli (sufficient area inside patch)
                qualifying = [
                    item for item in intersecting_annots
                    if item[2] >= self.min_glom_overlap_ratio
                ]

                candidate_info = {
                    "x0": x0,
                    "y0": y0,
                    "tissue_ratio": tissue_ratio,
                    "all_intersections": intersecting_annots,
                    "qualifying_gloms": qualifying,
                }

                if len(qualifying) > 0:
                    positive_candidates.append(candidate_info)
                elif len(intersecting_annots) == 0:
                    # Pure negative: tissue present, absolutely no glomerulus overlap
                    negative_candidates.append(candidate_info)

        return positive_candidates, negative_candidates

    def extract_patch_data(self, cand: Dict[str, Any], is_positive: bool) -> ExtractedPatch:
        """
        Read the region from WSI, downsample to target size, render masks and compute YOLO bboxes.
        """
        x0 = cand["x0"]
        y0 = cand["y0"]
        patch_id = f"{self.slide_id}_x{x0:06d}_y{y0:06d}"

        # 1. Read level 0 region
        reg = self.slide.read_region((x0, y0), 0, (self.patch_size_l0, self.patch_size_l0))
        img_rgb = np.array(reg.convert("RGB"))

        # 2. Resize to target patch size (20x)
        if self.downsample != 1.0:
            img_patch = cv2.resize(
                img_rgb,
                (self.patch_size, self.patch_size),
                interpolation=cv2.INTER_AREA
            )
        else:
            img_patch = img_rgb

        # 3. Create binary segmentation mask
        mask = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)
        yolo_bboxes: List[Tuple[int, float, float, float, float]] = []

        if is_positive:
            # Draw all intersecting polygon parts into the mask
            for annot, inter_geom, _ in cand["all_intersections"]:
                self._render_geometry_on_mask(inter_geom, x0, y0, mask)

            # Generate YOLO bounding boxes for qualifying glomeruli
            for annot, inter_geom, overlap_ratio in cand["qualifying_gloms"]:
                b_minx, b_miny, b_maxx, b_maxy = inter_geom.bounds

                # Convert to normalized YOLO coordinates [0, 1] relative to patch
                bw = (b_maxx - b_minx) / float(self.patch_size_l0)
                bh = (b_maxy - b_miny) / float(self.patch_size_l0)
                bcx = ((b_minx + b_maxx) / 2.0 - x0) / float(self.patch_size_l0)
                bcy = ((b_miny + b_maxy) / 2.0 - y0) / float(self.patch_size_l0)

                # Clamp values to [0.0, 1.0] for safety
                bcx = max(0.0, min(1.0, bcx))
                bcy = max(0.0, min(1.0, bcy))
                bw = max(0.0, min(1.0, bw))
                bh = max(0.0, min(1.0, bh))

                if bw > 0 and bh > 0:
                    yolo_bboxes.append((0, bcx, bcy, bw, bh))

        return ExtractedPatch(
            patch_id=patch_id,
            slide_id=self.slide_id,
            x0_l0=x0,
            y0_l0=y0,
            size_l0=self.patch_size_l0,
            patch_size=self.patch_size,
            image=img_patch,
            mask=mask,
            yolo_bboxes=yolo_bboxes,
            num_glomeruli=len(yolo_bboxes),
            tissue_ratio=cand["tissue_ratio"],
            is_positive=is_positive,
        )

    def _render_geometry_on_mask(
        self, geom: BaseGeometry, x0: int, y0: int, mask: np.ndarray
    ) -> None:
        """Helper to render Shapely Polygon or MultiPolygon onto the binary mask."""
        scale = 1.0 / self.downsample

        if isinstance(geom, Polygon):
            ext_pts = np.array(
                [((px - x0) * scale, (py - y0) * scale) for px, py in geom.exterior.coords],
                dtype=np.int32
            )
            cv2.fillPoly(mask, [ext_pts], 255)

            # Carve out interior holes if present
            for interior in geom.interiors:
                hole_pts = np.array(
                    [((px - x0) * scale, (py - y0) * scale) for px, py in interior.coords],
                    dtype=np.int32
                )
                cv2.fillPoly(mask, [hole_pts], 0)

        elif isinstance(geom, MultiPolygon):
            for p in geom.geoms:
                self._render_geometry_on_mask(p, x0, y0, mask)
        elif hasattr(geom, "geoms"):
            for sub_geom in getattr(geom, "geoms"):
                if isinstance(sub_geom, (Polygon, MultiPolygon)):
                    self._render_geometry_on_mask(sub_geom, x0, y0, mask)

    def extract_all(self, max_workers: int = 6) -> List[ExtractedPatch]:
        """
        Execute full extraction with balanced sampling of positive and negative patches
        using a multithreaded worker pool.
        """
        pos_cands, neg_cands = self.plan_patches()

        # Subsample negative patches according to ratio
        max_negs = int(len(pos_cands) * self.negative_sample_ratio)
        if len(neg_cands) > max_negs:
            selected_neg_indices = self.rng.choice(len(neg_cands), size=max_negs, replace=False)
            selected_neg_cands = [neg_cands[i] for i in selected_neg_indices]
        else:
            selected_neg_cands = neg_cands

        tasks = [(cand, True) for cand in pos_cands] + [(cand, False) for cand in selected_neg_cands]

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            extracted = list(executor.map(lambda t: self.extract_patch_data(t[0], t[1]), tasks))

        return extracted
