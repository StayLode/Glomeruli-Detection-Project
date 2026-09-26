"""
Parser for ASAP (Automated Slide Analysis Platform) XML annotation files.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union
import xml.etree.ElementTree as ET

from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.validation import make_valid


@dataclass
class ASAPAnnotation:
    """Represents a single annotation from an ASAP XML file."""
    name: str
    type_str: str
    group: str
    color: str
    geometry: BaseGeometry
    area: float
    bounds: Tuple[float, float, float, float]  # minx, miny, maxx, maxy
    centroid: Tuple[float, float]               # (x, y)


def parse_asap_xml(xml_path: Union[str, Path]) -> List[ASAPAnnotation]:
    """
    Parse an ASAP XML file and return a list of ASAPAnnotation objects.

    Args:
        xml_path: Path to the XML file.

    Returns:
        List of ASAPAnnotation objects with valid Shapely geometries.
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {xml_path}")

    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    annotations: List[ASAPAnnotation] = []

    for ann_elem in root.findall(".//Annotation"):
        name = ann_elem.get("Name", "")
        type_str = ann_elem.get("Type", "Polygon")
        group = ann_elem.get("PartOfGroup", "None")
        color = ann_elem.get("Color", "#F4FA58")

        coord_elems = ann_elem.findall(".//Coordinate")
        points = []
        for c in coord_elems:
            try:
                x = float(c.get("X", "0"))
                y = float(c.get("Y", "0"))
                points.append((x, y))
            except ValueError:
                continue

        if len(points) < 3:
            # Degenerate annotation (less than 3 vertices)
            continue

        poly = Polygon(points)
        if not poly.is_valid:
            poly = make_valid(poly)

        if poly.is_empty or poly.area <= 0:
            continue

        centroid = (poly.centroid.x, poly.centroid.y)
        bounds = poly.bounds  # (minx, miny, maxx, maxy)

        annotations.append(
            ASAPAnnotation(
                name=name,
                type_str=type_str,
                group=group,
                color=color,
                geometry=poly,
                area=float(poly.area),
                bounds=(float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])),
                centroid=(float(centroid[0]), float(centroid[1])),
            )
        )

    return annotations
