import os
import cv2
import numpy as np
import supervision as sv

from path_config import HOME
from loadmodel import sam_predictor
from segment_anything import SamPredictor

#generate segmentation masks
def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)



def segment_instances(SOURCE_IMAGE_PATH, image, detections, CLASSES, CONFIDENCE_THRESHOLD = 0.40):
    # Define confidence threshold (80%)
    
    
    # Get base image name
    image_name = os.path.basename(SOURCE_IMAGE_PATH) 

    # Define output dir for segmentation
    OUTPUT_DIR = f"{HOME}/instance_segmentation"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_IMAGE_PATH = os.path.join(OUTPUT_DIR, f"segmented_{image_name}")

    # Skip if no detections or invalid detections
    if detections is None or not hasattr(detections, 'confidence'):
        print("No valid detections provided for segmentation")
        return None

    # Filter detections by confidence threshold
    high_confidence_mask = np.array([
        conf >= CONFIDENCE_THRESHOLD 
        for conf in detections.confidence
        if conf is not None
    ])

    # Create filtered detections
    filtered_detections = sv.Detections(
        xyxy=detections.xyxy[high_confidence_mask],
        confidence=detections.confidence[high_confidence_mask],
        class_id=detections.class_id[high_confidence_mask],
        mask=None  # Will be populated later
    )

    # Only proceed if we have high-confidence detections
    if len(filtered_detections) == 0:
        print("No detections meet the 80% confidence threshold for segmentation")
        return None

    # Generate segmentation masks only for high-confidence detections
    filtered_detections.mask = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=filtered_detections.xyxy
    )

    # Create labels only for high-confidence detections
    labels = [
        f"{CLASSES[class_id]} {confidence:0.2f}"
        for class_id, confidence in zip(filtered_detections.class_id, filtered_detections.confidence)
        if class_id is not None and class_id < len(CLASSES)
    ]

    # Annotate image with masks and boxes
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    
    annotated_image = mask_annotator.annotate(
        scene=image.copy(), 
        detections=filtered_detections
    )
    annotated_image = box_annotator.annotate(
        scene=annotated_image, 
        detections=filtered_detections, 
        labels=labels
    )

    # Save the segmented image
    cv2.imwrite(OUTPUT_IMAGE_PATH, annotated_image)
    print(f"Segmented image saved at {OUTPUT_IMAGE_PATH} (showing only ≥80% confidence detections)")

    return filtered_detections.mask