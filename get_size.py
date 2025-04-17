import os
import cv2

def get_image_dimensions_from_dir(dir_path):
    """
    Returns the width and height of any image in the directory (all images are assumed to be of same dimensions).

    Parameters:
    - dir_path (str): Path to the directory containing images

    Returns:
    - (int, int): Width and height of the image
    """
    # List all files in the directory
    files = os.listdir(dir_path)
    
    # Filter out non-image files (basic check)
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if not image_files:
        raise ValueError("No image files found in the directory.")
    
    # Use the first image
    image_path = os.path.join(dir_path, image_files[0])
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error reading image: {image_path}")
    
    height, width = img.shape[:2]
    return width, height


def get_fps(video_path):
    """
    Returns the frames per second (fps) of the video.

    Parameters:
    - video_path (str): Path to the video file

    Returns:
    - float: Frames per second of the video
    """
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps