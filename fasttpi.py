
import os
import time
import shutil
import logging
import asyncio
import re
import base64

import json
from datetime import datetime
from uuid import uuid4
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from main import main_func  # Import your video processing function
from get_size import get_image_dimensions_from_dir
from get_size import get_fps
from zipfile import ZipFile

# bBox = []

# Directories
UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"
PROGRESS_DIR = "progress"
LOGS_DIR = "logs"
INTERRUPT_DIR = "interrupts"
NO_OF_OBJECTS = "no_of_objects"
ANNOTATED_FRAME_DIR = "segmented_frames_annotated"
PROCESSED_IMAGE_DIR = "processed_img"

for directory in [UPLOAD_DIR, PROCESSED_DIR, PROGRESS_DIR, LOGS_DIR, PROCESSED_IMAGE_DIR, ANNOTATED_FRAME_DIR]:
    os.makedirs(directory, exist_ok=True)
# Dictionary to store task statuses
tasks = {}

# ------------------------ Logging Setup ------------------------
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE = os.path.join(LOGS_DIR, f"logs_{timestamp}.log")

logger = logging.getLogger("FastAPI-App")
logger.setLevel(logging.DEBUG)

if logger.hasHandlers():
    logger.handlers.clear()

file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# ------------------------ Utility Functions ------------------------
def contains_status(text: str) -> bool:
    """
    Returns True if text contains "Processing..." or a percentage message like "50% completed".
    """
    pattern = r'Processing\.{3}|\b(?:[0-9]|[1-9][0-9])% completed\b'
    return bool(re.search(pattern, text))

def encode_frame_to_base64(frame_path: str) -> str:
    """
    Encodes an image frame to Base64 format.
    """
    if os.path.exists(frame_path):
        with open(frame_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    return ""

async def send_frames(file_id: str, websocket: WebSocket):
    """
    Continuously sends frames to the client while processing is ongoing.
    """
    global paused, resumed
    if paused==True:
        await websocket.send_json({
            "action": "PAUSE",
            "file_id": file_id
            # "frame_data": "paused"
        })
        logger.info("PAUSE send to client")
        return
    elif resumed==True:
        await websocket.send_json({
            "action": "RESUME",
            "file_id": file_id
        })
        logger.info("RESUME send to client")
    logger.info(f"Frame streaming started for {file_id}")
    frame_index = 0
    while tasks.get(file_id, "Not started") == "Processing...":
    # while re.search(r'\bProcessing\b', tasks.get(file_id, "Not started")):
        # frame_path = os.path.join("display_mask", f"{file_id}_frame_{frame_index}.png")
        frame_path = 'display_mask/5_plot_0_.png'
        
        if os.path.exists(frame_path):
            frame_data = encode_frame_to_base64(frame_path)
            await websocket.send_json({
                "action": "frame",
                "frame_data": frame_data,
                "file_id": file_id 
            })
            frame_index += 1
        else:
            logger.warning(f"Frame {frame_path} not found, waiting...")
        
        await asyncio.sleep(0.1)  # Adjust frame sending rate

    logger.info(f"Frame streaming ended for {file_id}")


async def async_track_progress(file_id: str, websocket: WebSocket):
    """
    Asynchronously monitor and send progress updates to the client.
    """
    logger.info(f"Progress tracking started for {file_id}")
    progress_file = os.path.join(PROGRESS_DIR, f"progress_{file_id}.txt")
    last_progress = 0
    # Continue until we hit or exceed 100
    while last_progress < 100:
        if os.path.exists(progress_file):
            try:
                with open(progress_file, "r") as f:
                    progress = int(f.read().strip())
            except ValueError:
                progress = last_progress  # Retain last known progress if conversion fails
            if progress > last_progress:
                last_progress = progress
                await websocket.send_json({
                    "action": "progress",
                    "file_id": file_id,
                    "progress": progress
                })
        await asyncio.sleep(2)
    await websocket.send_json({
        "action": "progress",
        "file_id": file_id,
        "progress": 100
    })

async def async_long_video_processing(file_id: str, input_classes: List[str], desired_fps: int,confidence_threshold:int, toggle_flag: bool, websocket: WebSocket):
    """
    Runs the video processing function asynchronously and sends updates over the websocket.
    """
    logger.info(f"Processing started for {file_id} with classes: {input_classes} at {desired_fps} FPS and Toggle is {toggle_flag}")
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp4")
    output_path = os.path.join(PROCESSED_DIR, f"{file_id}_processed.mp4")

    if not os.path.exists(input_path):
        logger.error(f"File {file_id} not found. Aborting processing.")
        tasks[file_id] = "Failed - File not found"
        await websocket.send_json({
            "action": "error",
            "file_id": file_id,
            "message": f"File {file_id} not found"
        })
        return

    tasks[file_id] = "Processing..."
    # Write an interrupt file to mark start
    with open(os.path.join(INTERRUPT_DIR, f"interrupt_{file_id}.txt"), "w") as f:
        f.write("Started\n")

    # Start tracking progress in the background
    progress_task = asyncio.create_task(async_track_progress(file_id, websocket))

    try:
        # Run the blocking main_func in a separate thread using asyncio.to_thread
        await asyncio.to_thread(main_func, input_path, output_path, input_classes, file_id, desired_fps,confidence_threshold, toggle_flag)
        tasks[file_id] = "Completed"
        logger.info(f"Processing completed successfully for {file_id}")
        await websocket.send_json({
            "action": "completed",
            "file_id": file_id,
            "message": "Processing completed"
        })
        logger.info('completed processing send to client from server')
    except Exception as e:
        tasks[file_id] = f"Failed - {str(e)}"
        logger.error(f"Error processing {file_id}: {str(e)}")
        await websocket.send_json({
            "action": "error",
            "file_id": file_id,
            "message": f"Error processing: {str(e)}"
        })
    await progress_task

# ------------------------ FastAPI WebSocket App ------------------------
# app = FastAPI()
app = APIRouter()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global bBox, paused, resumed
    await websocket.accept()
    # Use an asyncio lock to ensure that only one video is processed at a time.
    processing_lock = asyncio.Lock()
    paused = False
    resumed = False
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({"action": "error", "message": "Invalid JSON"})
                continue

            action = message.get("action")
            
            # ------------------------ Upload ------------------------
            if action == "upload":
                # Expect file_data as a base64 encoded string.
                file_data = message.get("file_data")
                if not file_data:
                    await websocket.send_json({
                        "action": "error",
                        "message": "No file data provided for upload"
                    })
                    continue
                file_id = str(uuid4())
                file_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp4")
                try:
                    with open(file_path, "wb") as f:
                        f.write(base64.b64decode(file_data))
                    logger.info(f"File uploaded successfully: {file_id}")
                    
                    fps = get_fps(file_path)
                    await websocket.send_json({
                        "action": "upload",
                        "file_id": file_id,
                        "message": "File uploaded successfully",
                        "fps": fps
                    })
                except Exception as e:
                    logger.error(f"Upload failed: {str(e)}")
                    await websocket.send_json({
                        "action": "error",
                        "message": "File upload failed"
                    })
            
            # ------------------------ Process ------------------------
            elif action == "process":
                file_id = message.get("file_id")
                classes_str = message.get("classes")
                desired_fps = message.get("desired_fps")
                confidence_threshold = message.get("confidence_threshold")
                toggle_flag = message.get("toggle")
                logger.info(f'toggle received on server side -> {toggle_flag}')

                if not file_id or not classes_str or desired_fps is None:
                    await websocket.send_json({
                        "action": "error",
                        "message": "Missing parameters for processing"
                    })
                    continue
                input_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp4")
                if not os.path.exists(input_path):
                    await websocket.send_json({
                        "action": "error",
                        "message": f"File {file_id} not found"
                    })
                    continue

                if processing_lock.locked():
                    await websocket.send_json({
                        "action": "error",
                        "message": "Another video is already being processed"
                    })
                    continue

                input_classes = [cls.strip() for cls in classes_str.split(",") if cls.strip()]
                logger.info(f"Processing request received for {file_id} with classes: {input_classes}")
                # Acquire lock and start processing
                async with processing_lock:
                    asyncio.create_task(
                        async_long_video_processing(file_id, input_classes, desired_fps,confidence_threshold, toggle_flag, websocket)
                    )
                    logger.info('before sleep')
                    time.sleep(5)
                    logger.info('after sleep')

                    # file_path_2_data = os.path.join("data")

                    # Width, Height = get_image_dimensions_from_dir(file_path_2_data)
                    # Width = str(Width)
                    # Height = str(Height)
                    # logger.info(f'width is: {Width} | height is: {Height}')
                    await websocket.send_json({
                        "action": "process",
                        "file_id": file_id,
                        "message": "Processing started",
                        "input_classes": input_classes,
                        "fps": desired_fps,
                        "confidence_threshold": confidence_threshold
                        # "width": Width,
                        # "height": Height
                    })

            # ------------------------ Interrupt ------------------------
            elif action == "interrupt":
                file_id = message.get("file_id")
                frame_num = message.get("frame_num")
                STOP = message.get("STOP")
                IOU = message.get("IOU")
                progress_file = os.path.join(PROGRESS_DIR, f"progress_{file_id}.txt")
                if os.path.exists(progress_file):
                    if file_id in tasks and contains_status(tasks[file_id]):
                        os.makedirs(INTERRUPT_DIR, exist_ok=True)
                        interrupt_file = os.path.join(INTERRUPT_DIR, f"interrupt_{file_id}.txt")
                        valid_stop_values = {"STOPPROCESS", "STOPDETECTION"}
                        content = STOP if STOP in valid_stop_values else "Interrupted"
                        with open(interrupt_file, "w") as f:
                            f.write(content + '\n')
                        logger.info(f"Interrupt successful for {file_id} with status: {content}")
                        with open(os.path.join(NO_OF_OBJECTS, f"no_of_objects_{file_id}.txt"), "r") as file:
                            first_line = file.readline().strip()
                            # print(first_line)
                            logger.info(f"number of objects -> {first_line}")
                        if content == "Interrupted":
                            temp = content
                            while temp != "Interruptedd":
                                with open(interrupt_file,"r") as f:
                                    temp = f.readline().strip()
                                    logger.info(f"temp -> {temp}")
                                time.sleep(2)  # Wait for 2 seconds
                                logger.info('inside while')
                        await websocket.send_json({
                            "action": "interrupt",
                            "file_id": file_id,
                            "message": "Processing interrupted",
                            "frame_num": frame_num,
                            "status": content,
                            "objects_present": first_line
                        })
                        logger.info('interrupt response send to client')
                    else:
                        logger.warning(f"Interrupt failed: {file_id} is not currently being processed")
                        await websocket.send_json({
                            "action": "error",
                            "message": "Video is not being processed"
                        })
                else:
                    logger.warning(f"Interrupt failed: Process {file_id} not started")
                    await websocket.send_json({
                        "action": "error",
                        "message": "Process not started"
                    })

            # ------------------------ Status ------------------------
            elif action == "status":
                file_id = message.get("file_id")
                progress_file = os.path.join(PROGRESS_DIR, f"progress_{file_id}.txt")
                if os.path.exists(progress_file):
                    with open(progress_file, "r") as f:
                        progress = f.read().strip()
                    if progress == "99":
                        await websocket.send_json({
                            "action": "status",
                            "file_id": file_id,
                            "status": "Completed"
                        })
                    else:
                        await websocket.send_json({
                            "action": "status",
                            "file_id": file_id,
                            "status": f"Processing {progress}%"
                        })
                else:
                    await websocket.send_json({
                        "action": "status",
                        "file_id": file_id,
                        "status": tasks.get(file_id, "Not started")
                    })

            # ------------------------ Download ------------------------
            elif action == "download":
                file_id = message.get("file_id")
                file_path = os.path.join(PROCESSED_DIR, f"{file_id}_processed.mp4")
                if os.path.exists(file_path):
                    with open(file_path, "rb") as f:
                        file_content = f.read()
                    file_base64 = base64.b64encode(file_content).decode("utf-8")
                    logger.info(f"Download request for {file_id}")
                    await websocket.send_json({
                        "action": "download",
                        "file_id": file_id,
                        "file_data": file_base64,
                        "message": "File downloaded successfully"
                    })
                else:
                    logger.error(f"Download failed: Processed file {file_id} not found.")
                    await websocket.send_json({
                        "action": "error",
                        "message": "Processed file not found"
                    })

            # ------------------------ Start Streaming Action ------------------------
            elif action == "start_streaming":
                asyncio.create_task(send_frames(file_id, websocket))
                logger.info('start_streaming received from client')


            # # ------------------------ GET BOX Action ------------------------
            # elif action == "bBox":
            #     bBox = message.get("bBox")

            # ------------------------ PAUSE Streaming Action ------------------------
            elif action == "PAUSE":
                print("[SERVER] Pausing video stream...")
                paused = True
                resumed = False
                logger.info('PAUSE received from Client')

            # ------------------------ RESUME Streaming Action ------------------------
            elif action == "RESUME":
                logger.info("[SERVER] Resuming video stream...")
                bBox = message.get("bBox")
                file_id = message.get("file_id")
                # save coordinates to file here...
                # bBox = str(bBox)
                logger.info(type(bBox))
                with open(os.path.join(NO_OF_OBJECTS, f"no_of_objects_{file_id}.txt"), "a") as file:
                    file.write(bBox+'\n')
                    logger.info('bBox written to file successfully')

                paused = False
                resumed = True
                
            # ------------------------ Unknown Action ------------------------

            else:
                await websocket.send_json({
                    "action": "error",
                    "message": "Unknown action"
                })
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")


# ------------------------ Download Processed ZIP File------------------------
@app.get("/download/images/{file_id}")
async def download_folder(file_id: str):
    """ Allows users to download the processed video file """
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


@app.get("/download_aug/{file_id}/{type_aug}")
async def download_augmented(file_id: str, type_aug: int):
    """ Allows users to download the augmented images """
    aug_zip_path = os.path.join(f"augmented/{type_aug}", f"augmented_{file_id}.zip")

    if os.path.exists(aug_zip_path):
        logger.info(f"Download request for augmented images: {file_id}")
        return FileResponse(aug_zip_path, media_type="application/zip", filename=f"augmented_{file_id}.zip")
    
    logger.error(f"Download failed: Augmented images {file_id} not found.")
    return JSONResponse(content={"error": "Augmented images not found"}, status_code=404)

# ------------------------ Run FastAPI Server ------------------------
if __name__ == "__main__":
    import uvicorn
    # logging.info("Starting FastAPI server...")
    # uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
    uvicorn.run(app, host="0.0.0.0", port=8001)
