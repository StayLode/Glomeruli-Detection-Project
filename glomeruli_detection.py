import os
import sys
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from PIL import Image, ImageStat
from tqdm import tqdm

# Disabilitiamo i noiosi warning di TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# 0A. Set TensorFlow Memory Growth
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e: print(e)

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import openslide

# --- IMPOSTAZIONI FISSE ---
PATCH_SIZE = 512
STRIDE = 256
CONFIDENCE_THRESHOLD = 0.7
BATCH_SIZE = 8  # Quante immagini processare in parallelo nella GPU

# !!! ATTENZIONE: INSERIRE QUI IL MAPPING ESATTO GENERATO DAL NOTEBOOK 02 !!!
CLUSTER_MAPPING = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5} 

def get_mask_rcnn_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model

def resize_with_padding(pil_img, expected_size=(224, 224)):
    pil_img.thumbnail(expected_size, Image.Resampling.LANCZOS)
    background = Image.new("RGB", expected_size, (255, 255, 255))
    offset_x = (expected_size[0] - pil_img.size[0]) // 2
    offset_y = (expected_size[1] - pil_img.size[1]) // 2
    background.paste(pil_img, (offset_x, offset_y))
    return background

def main(wsi_path, output_dir, models_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n[+] Loading Models...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"    Using device for Detection: {device}")

    # Load Mask R-CNN
    mask_rcnn = get_mask_rcnn_model(2)
    mask_rcnn.load_state_dict(torch.load(os.path.join(models_dir, 'identification/mask_rcnn_glomeruli.pth'), map_location=device))
    mask_rcnn.to(device)
    mask_rcnn.eval()

    # Load ResNet50
    feature_extractor = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')

    # Load PCA & KMeans
    with open(os.path.join(models_dir, "clustering/pca_model.pkl"), 'rb') as f:
        pca = pickle.load(f)
    with open(os.path.join(models_dir, "clustering/kmeans_model.pkl"), 'rb') as f:
        kmeans = pickle.load(f)

    # --- PHASE 1: DETECTION ---
    print(f"\n[+] Phase 1: Scanning WSI '{os.path.basename(wsi_path)}' for Glomeruli...")
    slide = openslide.OpenSlide(wsi_path)
    wsi_width, wsi_height = slide.dimensions
    
    raw_boxes = []
    raw_scores = []

    # Calcoliamo tutti gli step
    x_steps = list(range(0, wsi_width - PATCH_SIZE + 1, STRIDE))
    y_steps = list(range(0, wsi_height - PATCH_SIZE + 1, STRIDE))
    total_steps = len(y_steps) * len(x_steps)

    # Variabili per gestire il batching
    batch_tensors = []
    batch_coords = []

    with torch.no_grad():
        with tqdm(total=total_steps, desc="Sliding Window") as pbar:
            for y in y_steps:
                for x in x_steps:
                    
                    # 1. Lettura
                    patch = slide.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE)).convert("RGB")
                    
                    # 2. White Space Skipping (Salta se il patch è vetro bianco > 240/255)
                    gray_patch = patch.convert("L")
                    mean_brightness = ImageStat.Stat(gray_patch).mean[0]
                    if mean_brightness > 240:
                        pbar.update(1)
                        continue
                    
                    # 3. Preparazione per il Batch
                    img_tensor = F.to_tensor(patch)
                    batch_tensors.append(img_tensor)
                    batch_coords.append((x, y))
                    
                    # Se il batch è pieno o siamo all'ultimissimo step della WSI
                    is_last_step = (x == x_steps[-1] and y == y_steps[-1])
                    if len(batch_tensors) == BATCH_SIZE or is_last_step:
                        if len(batch_tensors) > 0:
                            # Stack per creare il tensore parallelo
                            batch_stack = torch.stack(batch_tensors).to(device)
                            
                            # Inferenza in un colpo solo!
                            predictions = mask_rcnn(batch_stack)
                            
                            # Traslazione coordinate per ogni immagine del batch
                            for idx, prediction in enumerate(predictions):
                                patch_x, patch_y = batch_coords[idx]
                                
                                for i in range(len(prediction['scores'])):
                                    score = prediction['scores'][i].item()
                                    if score >= CONFIDENCE_THRESHOLD:
                                        box = prediction['boxes'][i].cpu().numpy()
                                        raw_boxes.append([box[0] + patch_x, box[1] + patch_y, box[2] + patch_x, box[3] + patch_y])
                                        raw_scores.append(score)
                            
                            # Svuota il batch
                            batch_tensors = []
                            batch_coords = []
                    
                    pbar.update(1)

    # Non-Maximum Suppression (Rimuove i cloni)
    if len(raw_boxes) > 0:
        boxes_tensor = torch.tensor(raw_boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(raw_scores, dtype=torch.float32)
        keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold=0.3)
        final_boxes = boxes_tensor[keep_indices].numpy()
        print(f"    -> Found {len(final_boxes)} unique glomeruli.")
    else:
        print("    -> No glomeruli detected. Exiting.")
        sys.exit(0)


    # --- PHASE 2: CLASSIFICATION ---
    print("\n[+] Phase 2: Grading Disease Stages...")
    disease_stages = []

    for box in tqdm(final_boxes, desc="Extracting & Grading"):
        xmin, ymin, xmax, ymax = map(int, box)
        w, h = xmax - xmin, ymax - ymin
        
        # Evita errori se w o h sono <= 0
        if w <= 0 or h <= 0:
            disease_stages.append(0)
            continue
            
        glom_patch = slide.read_region((xmin, ymin), 0, (w, h)).convert("RGB")
        padded_img = resize_with_padding(glom_patch)
        img_array = np.array(padded_img) / 255.0
        
        img_raw = (img_array * 255.0).astype(np.float32)
        img_preprocessed = preprocess_input(np.expand_dims(img_raw, axis=0))
        
        features = feature_extractor.predict(img_preprocessed, verbose=0)
        reduced_features = pca.transform(features)
        raw_cluster = kmeans.predict(reduced_features)[0]
        
        stage = CLUSTER_MAPPING[raw_cluster]
        disease_stages.append(stage)


    # --- PHASE 3: VISUALIZATION & SAVING ---
    print("\n[+] Phase 3: Generating and Saving Final Map...")
    plot_level = slide.level_count - 1 
    downsample_factor = slide.level_downsamples[plot_level]
    wsi_thumbnail = slide.read_region((0,0), plot_level, slide.level_dimensions[plot_level]).convert("RGB")

    fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    ax.imshow(wsi_thumbnail)

    colors = cm.plasma(np.linspace(0, 1, 6))

    for box, stage in zip(final_boxes, disease_stages):
        xmin, ymin, xmax, ymax = box / downsample_factor
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                 linewidth=2, edgecolor=colors[stage], facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin - 2, f"St.{stage}", color='black', fontsize=8, weight='bold',
                bbox=dict(facecolor=colors[stage], alpha=0.7, edgecolor='none', pad=1))

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=colors[i], lw=4, label=f'Stage {i}') for i in range(6)]
    ax.legend(handles=legend_elements, loc='upper right', title="Necrotization Level", fontsize=12)

    ax.set_title(f"Glomeruli Analysis: {os.path.basename(wsi_path)}", fontsize=18)
    ax.axis("off")

    output_filename = os.path.join(output_dir, os.path.basename(wsi_path).replace(".svs", "_graded.png"))
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Result successfully saved to: {output_filename}")
    
    slide.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Glomeruli Detection and Grading Pipeline.")
    parser.add_argument("wsi_path", type=str, help="Path to the input .svs WSI file")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save the resulting image")
    parser.add_argument("--models_dir", type=str, default="./models", help="Directory where models are stored")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.wsi_path):
        print(f"Error: The file '{args.wsi_path}' does not exist.")
        sys.exit(1)
        
    main(args.wsi_path, args.output_dir, args.models_dir)