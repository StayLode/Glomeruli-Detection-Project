"""
Unified dataset builder.

Orchestrates WSI processing across all slides and outputs dual datasets:
1. YOLO format (images/, labels/, data.yaml)
2. Semantic segmentation format (images/, masks/)
3. Manifest CSV and dataset summary JSON with zero data leakage guarantees.
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import logging
import cv2
import openslide
import pandas as pd
from tqdm import tqdm
import yaml

from src.preprocessing.patch_extractor import ExtractedPatch, PatchExtractor
from src.preprocessing.tissue_detector import TissueDetector
from src.utils.xml_parser import parse_asap_xml

logger = logging.getLogger(__name__)

# Type alias for slide pair: (slide_id, svs_path, xml_path, split, fold)
Tuple_Pair = Tuple[str, Path, Path, str, str]


class DatasetBuilder:
    """Builds and structures the unified training dataset from WSI slides."""

    def __init__(self, config_path: str = "configs/dataset_config.yaml") -> None:
        self.config_path = Path(config_path)
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg: Dict[str, Any] = yaml.safe_load(f)

        self.raw_data_dir = Path(self.cfg["raw_data_dir"])
        self.output_dir = Path(self.cfg["output_dir"])

        # Mapping from slide_id to split name ('train', 'val', 'test')
        self.slide_to_split: Dict[str, str] = {}
        for split_name, slide_list in self.cfg["splits"].items():
            for sid in slide_list:
                self.slide_to_split[sid] = split_name

        # Mapping from slide_id to fold name ('fold_0', 'fold_1', etc.)
        self.slide_to_fold: Dict[str, str] = {}
        if "folds" in self.cfg:
            for fold_name, slide_list in self.cfg["folds"].items():
                for sid in slide_list:
                    self.slide_to_fold[sid] = fold_name

    def setup_directories(self) -> Dict[str, Path]:
        """Create the target directory structure for YOLO and segmentation."""
        paths = {
            "root": self.output_dir,
            "yolo_root": self.output_dir / "yolo",
            "seg_root": self.output_dir / "segmentation",
        }

        for split in ("train", "val", "test"):
            paths[f"yolo_img_{split}"] = self.output_dir / "yolo" / "images" / split
            paths[f"yolo_lbl_{split}"] = self.output_dir / "yolo" / "labels" / split
            paths[f"seg_img_{split}"] = self.output_dir / "segmentation" / "images" / split
            paths[f"seg_mask_{split}"] = self.output_dir / "segmentation" / "masks" / split

        for p in paths.values():
            p.mkdir(parents=True, exist_ok=True)

        return paths

    def find_slide_pairs(self) -> List[Tuple[str, Path, Path, str, str]]:
        """Find matching pairs of .svs and .xml files in raw_data_dir."""
        svs_files = sorted(self.raw_data_dir.glob("*.svs"))
        pairs = []

        for svs in svs_files:
            sid = svs.stem
            xml = self.raw_data_dir / f"{sid}.xml"
            if not xml.exists():
                logger.warning(f"No XML found for slide {sid}, skipping.")
                continue

            split = self.slide_to_split.get(sid, "train")
            fold = self.slide_to_fold.get(sid, "none")
            pairs.append((sid, svs, xml, split, fold))

        return pairs

    def build(self) -> pd.DataFrame:
        """Run the end-to-end dataset extraction process."""
        paths = self.setup_directories()
        pairs = self.find_slide_pairs()

        logger.info(f"Starting dataset extraction for {len(pairs)} WSI slides...")

        records: List[Dict[str, Any]] = []
        stats = {
            "splits": defaultdict(lambda: {"patches": 0, "positives": 0, "negatives": 0, "glomeruli": 0}),
            "slides": {}
        }

        for sid, svs_path, xml_path, split, fold in tqdm(pairs, desc="Processing WSIs"):
            logger.info(f"Opening slide: {sid} (Split: {split}, Fold: {fold})")
            slide = openslide.OpenSlide(str(svs_path))

            # 1. Parse ground truth annotations
            annotations = parse_asap_xml(xml_path)

            # 2. Segment tissue on overview
            tissue_detector = TissueDetector(
                slide,
                thumbnail_size=1024,
                max_v_thresh=238,
            )

            # 3. Extract patches
            extractor = PatchExtractor(
                slide=slide,
                slide_id=sid,
                annotations=annotations,
                tissue_detector=tissue_detector,
                target_mag=self.cfg.get("target_mag", 20),
                base_mag=int(slide.properties.get("openslide.objective-power", 40)),
                patch_size=self.cfg.get("patch_size", 1024),
                stride=self.cfg.get("stride", 768),
                min_tissue_ratio=self.cfg.get("min_tissue_ratio", 0.15),
                min_glom_overlap_ratio=self.cfg.get("min_glom_overlap_ratio", 0.20),
                negative_sample_ratio=self.cfg.get("negative_sample_ratio", 1.0),
                random_seed=self.cfg.get("seed", 42),
            )

            patches: List[ExtractedPatch] = extractor.extract_all(max_workers=6)

            slide_stats = {"positives": 0, "negatives": 0, "glomeruli": 0}

            # 4. Save patches and labels
            for p in patches:
                img_filename = f"{p.patch_id}.jpg"
                mask_filename = f"{p.patch_id}.png"
                label_filename = f"{p.patch_id}.txt"

                yolo_img_path = paths[f"yolo_img_{split}"] / img_filename
                yolo_lbl_path = paths[f"yolo_lbl_{split}"] / label_filename
                seg_mask_path = paths[f"seg_mask_{split}"] / mask_filename
                seg_img_path = paths[f"seg_img_{split}"] / img_filename

                # Save RGB Image if not exists
                if not yolo_img_path.exists():
                    cv2.imwrite(
                        str(yolo_img_path),
                        cv2.cvtColor(p.image, cv2.COLOR_RGB2BGR),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                    )

                # Link or copy image to segmentation folder
                if not seg_img_path.exists():
                    try:
                        seg_img_path.symlink_to(yolo_img_path.resolve())
                    except OSError:
                        import shutil
                        shutil.copy2(yolo_img_path, seg_img_path)

                # Save Binary Mask if not exists
                if not seg_mask_path.exists():
                    cv2.imwrite(str(seg_mask_path), p.mask)

                # Save YOLO Text Labels if not exists
                if not yolo_lbl_path.exists():
                    with open(yolo_lbl_path, "w", encoding="utf-8") as lf:
                        for cls_id, cx, cy, bw, bh in p.yolo_bboxes:
                            lf.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

                # Accumulate statistics
                if p.is_positive:
                    slide_stats["positives"] += 1
                    slide_stats["glomeruli"] += p.num_glomeruli
                    stats["splits"][split]["positives"] += 1
                    stats["splits"][split]["glomeruli"] += p.num_glomeruli
                else:
                    slide_stats["negatives"] += 1
                    stats["splits"][split]["negatives"] += 1

                stats["splits"][split]["patches"] += 1

                records.append({
                    "patch_id": p.patch_id,
                    "slide_id": sid,
                    "split": split,
                    "fold": fold,
                    "x0_l0": p.x0_l0,
                    "y0_l0": p.y0_l0,
                    "size_l0": p.size_l0,
                    "patch_size": p.patch_size,
                    "is_positive": p.is_positive,
                    "num_glomeruli": p.num_glomeruli,
                    "tissue_ratio": round(p.tissue_ratio, 4),
                    "image_path": str(yolo_img_path.relative_to(self.output_dir)),
                    "mask_path": str(seg_mask_path.relative_to(self.output_dir)),
                    "label_path": str(yolo_lbl_path.relative_to(self.output_dir)),
                })

            stats["slides"][sid] = slide_stats
            slide.close()

        # 5. Create manifest DataFrame
        df_manifest = pd.DataFrame(records)
        manifest_path = self.output_dir / "manifest.csv"
        df_manifest.to_csv(manifest_path, index=False)
        logger.info(f"Saved dataset manifest to {manifest_path}")

        # 6. Create YOLO data.yaml
        yolo_yaml_path = paths["yolo_root"] / "data.yaml"
        yolo_data = {
            "path": str(paths["yolo_root"].resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {
                0: "glomerulus"
            }
        }
        with open(yolo_yaml_path, "w", encoding="utf-8") as yf:
            yaml.dump(yolo_data, yf, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved YOLO configuration to {yolo_yaml_path}")

        # 7. Save dataset summary JSON
        summary_path = self.output_dir / "dataset_summary.json"
        with open(summary_path, "w", encoding="utf-8") as sf:
            json.dump(stats, sf, indent=2)
        logger.info(f"Saved dataset summary to {summary_path}")

        return df_manifest
