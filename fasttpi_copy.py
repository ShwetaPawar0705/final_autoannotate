import os
import time
import shutil
import logging
import threading
from datetime import datetime
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
from uuid import uuid4
from typing import List
from main import main_func
from path_config import HOME
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from zipfile import ZipFile
from SAM_for_video import video_to_frames

processing_lock = threading.Lock()
router = APIRouter()

UPLOAD_DIR = f"{HOME}/uploads"
PROCESSED_DIR = f"{HOME}/processed"
PROGRESS_DIR = f"{HOME}/progress"
LOGS_DIR = f"{HOME}/logs"
INTERRUPT_DIR = f"{HOME}/interrupts"
ANNOTATED_FRAME_DIR = os.path.join(HOME, "segmented_frames_annotated")
PROCESSED_IMAGE_DIR = f"{HOME}/processed_img"

for directory in [UPLOAD_DIR, PROCESSED_DIR, PROGRESS_DIR, LOGS_DIR, PROCESSED_IMAGE_DIR, ANNOTATED_FRAME_DIR]:
    os.makedirs(directory, exist_ok=True)

tasks = {}

def contains_status(text):
    pattern = r'Processing\.{{3}}|\b(?:[0-9]|[1-9][0-9])% completed\b'
    return bool(re.search(pattern, text))

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE = os.path.join(LOGS_DIR, f"logs_{timestamp}.log")

logger = logging.getLogger("    [FastAPI-App]")
logger.setLevel(logging.DEBUG)
if logger.hasHandlers():
    logger.handlers.clear()
file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

@router.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    file_id = str(uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp4")
    frames_dir_path = os.path.join(HOME, "data")

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        fps, total_frames = video_to_frames(file_path, frames_dir_path)
        logger.info(f"File uploaded successfully: {file_id}, FPS: {fps}")

        return {
            "file_id": file_id,
            "fps": fps,
            "total_frames": int(total_frames),
            "message": "File uploaded successfully"
        }
    except Exception as e:
        logger.error(f"Upload or processing failed: {str(e)}")
        return JSONResponse(content={"error": "File upload or processing failed"}, status_code=500)

def track_progress(file_id: str):
    logger.info(f" Progress Tracking started for {file_id} ")
    progress_file = os.path.join(PROGRESS_DIR, f"progress_{file_id}.txt")
    with tqdm(total=100, desc=f"Processing {file_id}", unit="%", ncols=80) as pbar:
        last_progress = 0
        while last_progress < 100:
            if os.path.exists(progress_file):
                with open(progress_file, "r") as f:
                    try:
                        progress = int(f.read().strip())
                    except ValueError:
                        progress = last_progress
                if progress > last_progress:
                    pbar.update(progress - last_progress)
                    last_progress = progress
            time.sleep(2)

def long_video_processing(file_id: str, input_classes: List[str], desired_fps: int, confidence_threshold: float):
    logger.info(f" Processing started for {file_id} with classes: {input_classes} at {desired_fps} FPS, confidence threshold: {confidence_threshold}")
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp4")
    output_path = os.path.join(PROCESSED_DIR, f"{file_id}_processed.mp4")

    if not os.path.exists(input_path):
        logger.error(f" File {file_id} not found. Aborting processing.")
        tasks[file_id] = "Failed - File not found"
        return

    tasks[file_id] = "Processing..."
    os.makedirs(INTERRUPT_DIR, exist_ok=True)
    interrupt_file = os.path.join(INTERRUPT_DIR, f"interrupt_{file_id}.txt")
    with open(interrupt_file, "w") as f:
        f.write('Started\n')

    progress_thread = threading.Thread(target=track_progress, args=(file_id,))
    progress_thread.start()

    try:
        main_func(input_path, output_path, input_classes, file_id, desired_fps, confidence_threshold)
        tasks[file_id] = "Completed"
        logger.info(f" Processing completed successfully for {file_id}")
    except Exception as e:
        tasks[file_id] = f"Failed - {str(e)}"
        logger.error(f" Error processing {file_id}: {str(e)}")

    progress_thread.join()

@router.post("/process/")
async def process_video(file_id: str = Form(...), classes: str = Form(...), desired_fps: int = Form(...), confidence_threshold: float = Form(...)):
    if not file_id or not classes or desired_fps <= 0:
        raise HTTPException(status_code=400, detail="Invalid input values")

    interrupt_file = os.path.join(INTERRUPT_DIR, f"interrupt_{file_id}.txt")
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp4")

    if not os.path.exists(input_path):
        logger.error(f"Processing failed: File {file_id} not found.")
        return JSONResponse(content={"error": "File not found"}, status_code=404)

    if os.path.exists(interrupt_file):
        with open(interrupt_file, "r") as f:
            progress = f.read().strip()
        if progress == "Ended" and processing_lock.locked():
            processing_lock.release()

    if processing_lock.locked():
        logger.warning("Processing request rejected: Another video is currently being processed.")
        return JSONResponse(content={"error": "Another video is already being processed"}, status_code=400)

    input_classes = [cls.strip() for cls in classes.split(",") if cls.strip()]
    logger.info(f"Processing request received for {file_id} with classes: {input_classes} and threshold: {confidence_threshold}")

    def video_processing_wrapper():
        with processing_lock:
            long_video_processing(file_id, input_classes, desired_fps, confidence_threshold)

    threading.Thread(target=video_processing_wrapper, daemon=True).start()

    return {"file_id": file_id, "message": "Processing started", "input_classes": input_classes, "fps": desired_fps, "confidence_threshold": confidence_threshold}

@router.post("/interrupt/{STOP}")
async def interrupt_processing(file_id: str = Form(...), frame_num: int = Form(...), STOP: str = None, IOU: float = None):
    logger.info(f" Interrupt request received for {file_id} at frame {frame_num}")
    if IOU:
        print('IOU threshold set to: ', IOU)
    progress_file = os.path.join(PROGRESS_DIR, f"progress_{file_id}.txt")
    if os.path.exists(progress_file):
        if file_id in tasks and contains_status(tasks[file_id]):
            os.makedirs(INTERRUPT_DIR, exist_ok=True)
            interrupt_file = os.path.join(INTERRUPT_DIR, f"interrupt_{file_id}.txt")
            valid_stop_values = {"STOPPROCESS", "STOPDETECTION"}
            content = STOP if STOP in valid_stop_values else "Interrupted"
            with open(interrupt_file, "w") as f:
                f.write(content + '\n')
            logger.info(f" Interrupt successful for {file_id} with status: {content}")
            return {"file_id": file_id, "message": "Processing interrupted", "frame_num": frame_num, "status": content}
        else:
            logger.warning(f" Interrupt failed: {file_id} is not currently being processed")
            return JSONResponse(content={"error": "Video is not being processed"}, status_code=400)
    logger.warning(f" Interrupt failed: Process {file_id} not started")
    return JSONResponse(content={"error": "Process not started"}, status_code=404)

@router.get("/status/{file_id}")
async def get_status(file_id: str):
    progress_file = os.path.join(PROGRESS_DIR, f"progress_{file_id}.txt")
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            progress = f.read().strip()
        if progress == "99":
            logger.info(f" Status check for {file_id}: Completed")
            return {"file_id": file_id, "status": "Completed"}
        logger.info(f" Status check for {file_id}: {progress}% completed")
        return {"file_id": file_id, "status": f"Processing {progress}%"}
    logger.info(f" Status check for {file_id}: {tasks.get(file_id, 'Not started')}")
    return {"file_id": file_id, "status": tasks.get(file_id, "Not started")}

@router.get("/download/{file_id}")
async def download_video(file_id: str):
    file_path = os.path.join(PROCESSED_DIR, f"{file_id}_processed.mp4")
    if os.path.exists(file_path):
        logger.info(f" Download request for {file_id}")
        return FileResponse(file_path, media_type="video/mp4", filename=f"{file_id}_processed.mp4")
    logger.error(f" Download failed: Processed file {file_id} not found.")
    return JSONResponse(content={"error": "Processed file not found"}, status_code=404)

@router.get("/download/images/{file_id}")
async def download_annotated_images(file_id: str):
    annotated_dir = os.path.join(ANNOTATED_FRAME_DIR, file_id)
    zip_path = os.path.join(PROCESSED_IMAGE_DIR, f"{file_id}_annotated_images.zip")

    if not os.path.isdir(annotated_dir):
        logger.error(f"Annotated image folder not found for {file_id}")
        return JSONResponse(content={"error": "Annotated image folder not found"}, status_code=404)

    if os.path.exists(zip_path):
        logger.info(f"Sending cached zip for annotated images: {file_id}")
        return FileResponse(zip_path, media_type="application/zip", filename=f"{file_id}_annotated_images.zip")

    image_files = [f for f in os.listdir(annotated_dir) if f.endswith(".jpg")]
    if not image_files:
        logger.warning(f"No annotated images found for {file_id}")
        return JSONResponse(content={"error": "No annotated images to zip"}, status_code=404)

    try:
        with ZipFile(zip_path, "w") as zipf:
            for img_file in image_files:
                full_path = os.path.join(annotated_dir, img_file)
                zipf.write(full_path, arcname=img_file)
        logger.info(f"Annotated images zipped for {file_id}")
        return FileResponse(zip_path, media_type="application/zip", filename=f"{file_id}_annotated_images.zip")
    except Exception as e:
        logger.error(f"Error zipping annotated images for {file_id}: {e}")
        return JSONResponse(content={"error": "Failed to zip annotated images"}, status_code=500)
