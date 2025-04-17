import os
import cv2
import numpy as np

def plot_and_save_masks(detections, CLASSES, output_dir, image_name):
    # Ensure main output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create a subdirectory for this image inside segment_masks/
    image_base_name = os.path.splitext(image_name)[0]  # Remove file extension
    image_output_dir = os.path.join(output_dir, image_base_name)
    os.makedirs(image_output_dir, exist_ok=True)

    # Check if detections contain masks
    if not hasattr(detections, "mask") or len(detections.mask) == 0:
        print(f"Warning: No masks found for {image_name}. Skipping save.")
        return

    # Loop through each mask and save it with class name
    for idx, mask in enumerate(detections.mask):
        class_id = detections.class_id[idx] if idx < len(detections.class_id) else None
        class_name = CLASSES[class_id] if class_id is not None and class_id < len(CLASSES) else "unknown"

        # Define save path with class name
        mask_name = f"{class_name}_{idx}.png"
        save_path = os.path.join(image_output_dir, mask_name)

        # Convert mask to uint8 format (0-255) for saving
        mask_uint8 = (mask * 255).astype(np.uint8)

        # Save the mask using OpenCV
        cv2.imwrite(save_path, mask_uint8)

        print(f"Saved mask {idx} ({class_name}) at: {save_path}")

    print(f"All masks saved successfully for {image_name} in {image_output_dir}.")
