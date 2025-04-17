# import os
# import cv2
# import shutil
# import imageio
# from imgaug import augmenters as iaa

# def augment_and_zip(dir_path, file_id):
#     """
#     Augments segmented images and saves them as a zip file in 'augmented/' directory.

#     Parameters:
#     - dir_path (str): Path to the directory containing segmented images.
#     - file_id (str): Unique ID for naming the output zip file.
#     """

#     # Create temp directory for augmented images
#     temp_aug_dir = f"temp_augmented_{file_id}"
#     if os.path.exists(temp_aug_dir):
#         shutil.rmtree(temp_aug_dir)
#     os.makedirs(temp_aug_dir)

#     # Define augmentation pipeline
#     seq = iaa.Sequential([
#         iaa.Fliplr(0.5),
#         iaa.Affine(rotate=(-25, 25)),
#         iaa.Multiply((0.8, 1.2)),
#         iaa.GaussianBlur(sigma=(0, 1.0))
#     ])

#     # Filter and augment images
#     image_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

#     for img_name in image_files:
#         img_path = os.path.join(dir_path, img_name)
#         image = cv2.imread(img_path)

#         if image is None:
#             print(f"Warning: Failed to read {img_name}")
#             continue

#         for i in range(3):  # 3 augmentations per image
#             aug_img = seq(image=image)
#             aug_filename = f"{os.path.splitext(img_name)[0]}_aug{i+1}.png"
#             out_path = os.path.join(temp_aug_dir, aug_filename)
#             imageio.imwrite(out_path, cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB))

#     # Create final augmented dir
#     final_output_dir = "augmented"
#     os.makedirs(final_output_dir, exist_ok=True)

#     # Create zip archive
#     zip_filename = f"augmented_{file_id}"
#     zip_path = os.path.join(final_output_dir, zip_filename)
#     shutil.make_archive(zip_path, 'zip', temp_aug_dir)

#     # Clean up temp folder
#     shutil.rmtree(temp_aug_dir)

#     print(f"✅ Augmented images saved and zipped at: {zip_path}.zip")

#####part 2
import os
import cv2
import shutil
import imageio
import albumentations as A
from albumentations.augmentations import functional as F
# from albumentations.augmentations.dropout.cutout import Cutout

def augment_and_zip(dir_path, file_id, aug_type=1):
    """
    Augments images based on aug_type and zips them into 'augmented/augmented_{file_id}.zip'.

    Parameters:
    - dir_path (str): Path to the segmented images.
    - file_id (str): ID to name the zip file.
    - aug_type (int): 1 for Geometric, 2 for Photometric, 3 for Synthetic augmentations.
    """

    temp_aug_dir = f"temp_augmented_{file_id}"
    if os.path.exists(temp_aug_dir):
        shutil.rmtree(temp_aug_dir)
    os.makedirs(temp_aug_dir)

    # Augmentation Pipelines
    if aug_type == 1:  # Geometric
        transform = A.Compose([
            A.HorizontalFlip(p=0.7),
            A.VerticalFlip(p=0.4),
            A.Rotate(limit=90, p=0.7),
            A.Affine(translate_percent=0.1, scale=(0.8, 1.2), rotate=(-45, 45), shear=(-10, 10), p=0.7),
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),
            A.GridDistortion(p=0.3),
        ])
    elif aug_type == 2:  # Photometric
        transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.4),
            A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.4),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            A.ChannelShuffle(p=0.3),
        ])
    elif aug_type == 3:  # Synthetic
        transform = A.Compose([
            # A.Cutout(num_holes=10, max_h_size=20, max_w_size=20, fill_value=0, p=0.7),
            A.CoarseDropout(max_holes=10, max_height=20, max_width=20, fill_value=0, p=0.7),
            A.GaussNoise(p=0.6),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=0.4),
            A.RandomRain(blur_value=3, brightness_coefficient=0.9, drop_width=1, drop_length=20, p=0.4),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
            A.Solarize(p=0.3),
        ])
    else:
        raise ValueError("aug_type must be 1 (Geometric), 2 (Photometric), or 3 (Synthetic)")

    image_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_name in image_files:
        img_path = os.path.join(dir_path, img_name)
        image = cv2.imread(img_path)

        if image is None:
            print(f"Skipping unreadable image: {img_name}")
            continue

        for i in range(5):  # Generate 5 diverse variants per image
            augmented = transform(image=image)
            aug_img = augmented['image']
            out_name = f"{os.path.splitext(img_name)[0]}_aug{i+1}.png"
            out_path = os.path.join(temp_aug_dir, out_name)
            imageio.imwrite(out_path, cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB))

    # Final Output Directory
    final_output_dir = f"augmented/{aug_type}"
    os.makedirs(final_output_dir, exist_ok=True)

    # Create Zip
    zip_filename = f"augmented_{file_id}"
    zip_path = os.path.join(final_output_dir, zip_filename)
    shutil.make_archive(zip_path, 'zip', temp_aug_dir)

    # Clean Up
    shutil.rmtree(temp_aug_dir)

    print(f"✅ {zip_filename}.zip created in 'augmented/{aug_type}/' directory.")


# augment_and_zip('instance_segmentation', 1234)
# augment_and_zip('instance_segmentation', 12345,2)
# augment_and_zip('instance_segmentation', 123456,3)

