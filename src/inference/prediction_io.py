"""Machine-readable post-NMS detections with explicit segmentation provenance."""
from pathlib import Path
from src.utils.evaluation import file_identity, write_json


def export_predictions(result, destination, yolo_weights, unet_weights):
    sizes=[len(result[key]) for key in ("detected_boxes_l0","detected_polygons_l0","detected_scores","patch_records")]
    if len(set(sizes)) != 1:
        raise ValueError("Post-NMS boxes, masks, scores and patch records must align")
    records=[]
    for i,(box,polygon,score,patch) in enumerate(zip(result["detected_boxes_l0"],result["detected_polygons_l0"],result["detected_scores"],result["patch_records"])):
        segmented=bool(result["cascade_mode"] and patch.get("patch_contour") is not None)
        records.append({"prediction_id":f"{result['slide_id']}#{i}","bbox_l0":box,"polygon_l0":polygon,
                        "score":score,"mask_source":"unet" if segmented else "bbox_fallback"})
    document={"schema_version":1,"stage":"wsi_predictions","slide_id":result["slide_id"],
              "svs_path":str(Path(result["svs_path"]).resolve()),"cascade_mode":result["cascade_mode"],
              "level0_dimensions":result["level0_dimensions"],"pixel_size_l0_um":result.get("pixel_size_l0_um"),
              "yolo_weights":file_identity(yolo_weights),"unet_weights":file_identity(unet_weights) if unet_weights else None,
              "detections":records,"detection_segmentation_metrics":result["evaluation"]}
    write_json(destination,document)
    return Path(destination)
