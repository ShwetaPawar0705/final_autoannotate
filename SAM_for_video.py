import os
import cv2
import shutil
import numpy as np
import supervision as sv
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from path_config import HOME
from loadmodel import sam2_model
from utils import show_mask, show_box
from notebooks.test_opencv_backup import annotate_image
from notebooks.mask_matching_backup import match_masks
from utils import show_mask, show_points
from new_line import monitor_file

# from fasttpi import bBox

FLAGG = []
DISPLAY_PATH=None
display_mask_dir = 'display_mask'
os.makedirs(display_mask_dir, exist_ok=True)

# get frames from videos
def video_to_frames(VIDEO_PATH, FRAMES_DIR_PATH):

    # capture the video
    cap = cv2.VideoCapture(VIDEO_PATH)
    # get fps and the number of frames
    fps = cap.get(cv2.CAP_PROP_FPS)
    totalNoFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    # initialize a scaling to 50%
    SCALE_FACTOR = 0.5

    # generate frames using supervision
    frames_generator = sv.get_video_frames_generator(VIDEO_PATH)
    # frames_generator = sv.get_video_frames_generator(VIDEO_PATH,stride=10)
    sink = sv.ImageSink(
        target_dir_path=FRAMES_DIR_PATH,
        image_name_pattern="{:05d}.jpeg")

    with sink:
        for frame in frames_generator:
            # scale image
            frame = sv.scale_image(frame, SCALE_FACTOR)
            # save image
            sink.save_image(frame)

    return fps, totalNoFrames


# EXAMPLE USAGE
# VIDEO_PATH=os.path.join(HOME, "video_data", "basketball_1.mp4")
FRAMES_DIR_PATH=os.path.join(HOME, "data")
# fps, totalNoFrames=video_to_frames(VIDEO_PATH, FRAMES_DIR_PATH)


# to add the segment masks to the model
def bbox_to_sam(file_id,Cust_Pt_Flag, skip_frames, SOURCE_DIR, source_image_path, FRAME_IDX, inference_index, detections, CLASSES, image, MASK):
    global FLAGG, DISPLAY_PATH, object_masks
    DISPLAY_PATH = source_image_path

    INTERRUPT_DIR = "interrupts"
    interrupt_file = os.path.join(INTERRUPT_DIR, f"interrupt_{file_id}.txt")

    NO_OF_OBJECTS = "no_of_objects"
    
    # Initialize MaskAnnotator
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.CLASS)

    # define inference frame dir
    FRAMES_DIR_PATH = os.path.join(HOME, "inference_frames")

    # save inference frames in dir (source img to int(fps/4))
    #___________________________________________________________
    # Create destination directory if it doesn't exist
    os.makedirs(FRAMES_DIR_PATH, exist_ok=True)

    # Get sorted list of images (numeric sorting)
    image_list = sorted(
        [f for f in os.listdir(SOURCE_DIR) if f.endswith(".jpeg")],
        key=lambda x: int(os.path.splitext(x)[0])  # Extract number and sort numerically
    )

    # Get the index of the input image
    input_image_name = os.path.basename(source_image_path)

    if input_image_name in image_list:
        start_index = image_list.index(input_image_name)  # Get image from input image
        with open(interrupt_file, "r") as f:
            interrupt = f.read().strip()
        if interrupt=="STOPDETECTION":
            end_index = max(start_index + skip_frames+1, len(image_list))  # Ensure within bounds
        else:
            end_index = min(start_index + skip_frames+1, len(image_list))  # Ensure within bounds

        # Copy the frames
        copied_images = []
        # Copy the next 10 images
        for i in range(start_index, end_index):
            src_path = os.path.join(SOURCE_DIR, image_list[i])
            dest_path = os.path.join(FRAMES_DIR_PATH, image_list[i])
            shutil.copy2(src_path, dest_path)  # Use shutil.move() if needed
            copied_images.append(dest_path)
            print(f"Copied: {image_list[i]}")

        print("Process completed!")
    else:
        print("Input image not found in the directory.")


    # Initialize inference state
    inference_state = sam2_model.init_state(FRAMES_DIR_PATH)
    print("Inference state initialized")

    
    # Remove the copied frames after inference
    for image_path in copied_images:
        try:
            os.remove(image_path)
            print(f"Deleted: {image_path}")
        except Exception as e:
            print(f"Error deleting {image_path}: {e}")

    print("Cleanup completed!")

    boxes_by_label = {label: [] for label in CLASSES}

    # Group bounding boxes by label
    for box, class_id in zip(detections.xyxy, detections.class_id):
        label = CLASSES[class_id]
        boxes_by_label[label].append(box)

    print(f"Boxes grouped by label: {boxes_by_label}")

    object_ids = []
    mask_logits = []
    object_id_counter = 1  # Start unique ID counter from 1



    # #____________________________________________________________
    # # get the point coordinates here....
    # if Cust_Pt_Flag==True:
    #     # Take a look at the first video frame
    #     frame_idx = 5
    #     # Call the function to annotate
    #     bboxes = annotate_image(DISPLAY_PATH if len(FLAGG)!=0 else source_image_path)
    #     # bboxes = annotate_image(source_image_path, boxes_by_label)


    #     # Print or use the bounding boxes
    #     print("Final Annotated Bounding Boxes from interrupt:", bboxes)

    #     points=[]
    #     labels=[]
    #     for index, cord in enumerate(bboxes):
    #         if cord[4] ==1:
    #             ann_obj_id=object_id_counter
    #             # points = np.array([[cord[0], cord[1]]], dtype=np.float32)
    #             points.append([cord[0], cord[1]])
    #             if index%2==0:
    #                 # labels = np.array([1], np.int32)
    #                 labels.append(1)
    #             else:
    #                 labels.append(0)

    #     points = np.array(points, dtype=np.float32)
    #     labels = np.array(labels, np.int32)

    #     ann_frame_idx = inference_index  # the frame index we interact with

    #     # Assign a unique object ID to each mask
    #     _, out_obj_ids, out_mask_logits = sam2_model.add_new_points(
    #         inference_state=inference_state,
    #         frame_idx=ann_frame_idx,
    #         obj_id=ann_obj_id,
    #         points=points,
    #         labels=labels,
    #     )

    #     # Extract the specific mask and process it
    #     selected_mask = out_mask_logits[object_id_counter - 1].cpu().numpy()
    #     # Apply a threshold to create a binary mask
    #     binary_mask = selected_mask > 0  # Adjust threshold if needed

    #     # Count nonzero pixels in the binary mask
    #     nonzero_count = np.count_nonzero(binary_mask)
                                
    #     FLAGG = (out_mask_logits[0] > 0.0).cpu().numpy()
    #     # FLAGG = np.expand_dims(selected_mask, axis=0)  # Add batch dimension if needed


    #     print('Added points:\n')
    #     print('mask shape - ',out_mask_logits.shape)

    #     print('out_obj_ids: ',out_obj_ids)

    #     # Print the desired output
    #     print(f"  - [{object_id_counter-1}] Mask shape: {selected_mask[object_id_counter-1].shape}, Nonzero pixels: {nonzero_count}")
    #     # print(f"  - [{object_id_counter-1}] Mask shape: {selected_mask.shape}, Nonzero pixels: {np.count_nonzero(selected_mask)}")


    #     # show the results on the current (interacted) frame
    #     plt.figure(figsize=(9, 6))
    #     plt.title(f"frame {ann_frame_idx}")
    #     plt.imshow(Image.open(source_image_path))
    #     # show_points(points, labels, plt.gca())
    #     show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

    #     # Save the plot to the directory
    #     save_path = os.path.join("tempp", f"{frame_idx}_plot_0_.png")
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save with high quality
    #     plt.close()  # Close the plot to free memory

    #     print(f"Plot saved at: {save_path}")
        
    #     print('interrupted code should resume here\n')

    #     ##___________________________________________________________________________
            



    # Iterate over all labels and their bounding boxes
    for label, boxes in boxes_by_label.items():
        if not boxes:  # Skip labels with no bounding boxes
            continue

        print(f"\nProcessing label: {label}")

        for box in boxes:  # Assign a new object ID for each bounding box
            x, y, w, h = map(int, box)
            print(f"  - Bounding box_before (x, y, w, h): ({x}, {y}, {w}, {h})")

            # Create mask
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            mask[y:h, x:w] = 255
            
            print(f"  - [{object_id_counter-1}] Mask shape: {MASK[object_id_counter-1].shape}, Nonzero pixels: {np.count_nonzero(MASK)}")
            # print(MASK)
            # print(MASK[object_id_counter-1])
            #below one 'mask' is the bounding boxes.
            # print(f"  - [_] Mask shape: {mask.shape}, Nonzero pixels: {np.count_nonzero(mask)}")

            if len(FLAGG)!=0:
                print('entered ->', len(FLAGG))
                MASK_FLAG = FLAGG
                print(f"  - [{object_id_counter-1}] Mask shape: {MASK[object_id_counter-1].shape}, Nonzero pixels: {np.count_nonzero(MASK)}")
                if MASK_FLAG.ndim == 4:
                    MASK_FLAG = np.squeeze(MASK_FLAG, axis=1)
                    print('squeezed mask shape: ', MASK_FLAG.shape)
                print('value of MASK changed: ', len(MASK_FLAG))
                # # Assign a unique object ID to each mask
                # _, obj_ids, mask_logit = sam2_model.add_new_mask(
                #     inference_state=inference_state,
                #     frame_idx=inference_index,
                #     obj_id=object_id_counter-1,  # Unique ID per bounding box
                #     mask=MASK[object_id_counter-1-1]
                # )
                # Call the function
                MASK = match_masks(MASK_FLAG, MASK)
                print('should exit loop here')
            # else:
            #     # Assign a unique object ID to each mask
            #     _, obj_ids, mask_logit = sam2_model.add_new_mask(
            #         inference_state=inference_state,
            #         frame_idx=inference_index,
            #         obj_id=object_id_counter,  # Unique ID per bounding box
            #         mask=MASK[object_id_counter-1]
            #     )

            # Assign a unique object ID to each mask
            _, obj_ids, mask_logit = sam2_model.add_new_mask(
                inference_state=inference_state,
                frame_idx=inference_index,
                obj_id=object_id_counter,  # Unique ID per bounding box
                mask=MASK[object_id_counter-1]
            )

            print('add_new_mask done right')
            # print('length_m: ', len(mask_logit))

            masks = (mask_logit > 0.0).cpu().numpy()
            if masks.ndim == 4:
                masks = np.squeeze(masks, axis=1)

            # convert masks to BB
            detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks=masks),
                mask=masks,
                class_id=np.array(obj_ids)
            )

            frame_path = source_image_path
            frame = cv2.imread(frame_path)

            # annotate box on the image
            annotated_frame = mask_annotator.annotate(scene=frame.copy(), detections=detections)

            DISPLAY_PATH = os.path.join(display_mask_dir, '5_plot_0_.png')
            cv2.imwrite(DISPLAY_PATH, annotated_frame)

            #____________________________________________________________
            # get the point coordinates here....
            if Cust_Pt_Flag==True:
                # Take a look at the first video frame
                frame_idx = 5
                # Call the function to annotate
                # bboxes = annotate_image(DISPLAY_PATH if len(FLAGG)!=0 else source_image_path)

                with open(interrupt_file, "w") as f:
                    f.write("Interruptedd")

                #here it should pause until the coordinates are received...from API action=bBox
                # bboxes = annotate_image(DISPLAY_PATH)
                bboxes = list(monitor_file(os.path.join(NO_OF_OBJECTS, f"no_of_objects_{file_id}.txt")))
                # bboxes = bBox
                
                # bboxes = annotate_image(source_image_path, boxes_by_label)


                # Print or use the bounding boxes
                print("Final Annotated Bounding Boxes from interrupt:", bboxes)

                points=[]
                labels=[]
                for index, cord in enumerate(bboxes):
                    if cord[4] ==1:
                        ann_obj_id=object_id_counter
                        # points = np.array([[cord[0], cord[1]]], dtype=np.float32)
                        points.append([cord[0], cord[1]])
                        if index%2==0:
                            # labels = np.array([1], np.int32)
                            labels.append(1)
                        else:
                            labels.append(0)

                points = np.array(points, dtype=np.float32)
                labels = np.array(labels, np.int32)

                print('labels: ', labels)

                ann_frame_idx = inference_index  # the frame index we interact with

                # Assign a unique object ID to each mask
                _, out_obj_ids, out_mask_logits = sam2_model.add_new_points(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    points=points,
                    labels=labels,
                )

                print('length_pt: ', len(out_mask_logits))
                # Extract the specific mask and process it
                selected_mask = out_mask_logits[object_id_counter - 1].cpu().numpy()
                # Apply a threshold to create a binary mask
                binary_mask = selected_mask > 0  # Adjust threshold if needed

                # Count nonzero pixels in the binary mask
                nonzero_count = np.count_nonzero(binary_mask)
                                        
                # FLAGG = (out_mask_logits > 0.0).cpu().numpy()
                # FLAGG = np.expand_dims(selected_mask, axis=0)  # Add batch dimension if needed


                print('Added points:\n')
                print('mask shape - ',out_mask_logits.shape)

                print('out_obj_ids: ',out_obj_ids)

                # Print the desired output
                # print(f"  - [{object_id_counter-1}] Mask shape: {selected_mask[object_id_counter-1].shape}, Nonzero pixels: {nonzero_count}")
                # print(f"  - [{object_id_counter-1}] Mask shape: {selected_mask.shape}, Nonzero pixels: {np.count_nonzero(selected_mask)}")


                # show the results on the current (interacted) frame
                plt.figure(figsize=(9, 6))
                plt.title(f"frame {ann_frame_idx}")
                plt.imshow(Image.open(source_image_path))
                # show_points(points, labels, plt.gca())
                show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

                # Save the plot to the directory
                save_path = os.path.join("tempp", f"{frame_idx}_plot_0_.png")
                plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save with high quality
                plt.close()  # Close the plot to free memory

                print(f"Plot saved at: {save_path}")
                
                print('interrupted code should resume here\n')

                ##___________________________________________________________________________
                    






            # masks = (mask_logit > 0.0).cpu().numpy()
            # if masks.ndim == 4:
            #     masks = np.squeeze(masks, axis=1)

            # # convert masks to BB
            # detections = sv.Detections(
            #     xyxy=sv.mask_to_xyxy(masks=masks),
            #     mask=masks,
            #     class_id=np.array(obj_ids)
            # )

            # frame_path = source_image_path
            # frame = cv2.imread(frame_path)

            # # annotate box on the image
            # annotated_frame = mask_annotator.annotate(scene=frame.copy(), detections=detections)


            
            # show the results on the current (interacted) frame
            plt.figure(figsize=(9, 6))
            plt.title(f"frame {FRAME_IDX}")
            plt.imshow(Image.open(f"{source_image_path}"))
            # show_box(box, plt.gca())
            show_mask((mask_logit[object_id_counter-1] > 0.0).cpu().numpy(), plt.gca(), obj_id=obj_ids[object_id_counter-1])
            
            # Save the plot to the directory
            save_path = os.path.join("bbox_output", f"{FRAME_IDX}_plot_{object_id_counter}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save with high quality
            # plt.savefig(DISPLAY_PATH, dpi=300, bbox_inches='tight')  # Save with high quality
            plt.close()  # Close the plot to free memory

            # DISPLAY_PATH = os.path.join(display_mask_dir, '5_plot_0_.png')
            # cv2.imwrite(DISPLAY_PATH, annotated_frame)


            print(f"Plot saved at: {save_path}")

            print(f"  - Assigned Object ID: {object_id_counter}")
            print(f"  - Object IDs returned: {obj_ids}")
            print(f"  - Mask logit shape: {mask_logit.shape if isinstance(mask_logit, np.ndarray) else type(mask_logit)}")

            # object_ids.extend(obj_ids)  # Collect object IDs
            object_ids=obj_ids
            # mask_logits.append(mask_logit)  # Collect mask logits
            mask_logits=mask_logit

            

            object_id_counter += 1  # Increment object ID for the next bounding box

    print(f"\nFinal Object IDs: {object_ids}")
    print(f"Final number of mask logits: {len(mask_logits)}")

    return object_ids, mask_logits, inference_state


# to pass the masks to the skip frames
def visualize_segment_video(index, SOURCE_VIDEO, inference_state,file_id, toggle_flag, labels):
    global FLAGG, DISPLAY_PATH, object_masks

    INTERRUPT_DIR = "interrupts"
    interrupt_file = os.path.join(INTERRUPT_DIR, f"interrupt_{file_id}.txt")

    SCALE_FACTOR = 1.0
    FRAMES_DIR_PATH = os.path.join(HOME, "data")
    SOURCE_FRAMES = Path(FRAMES_DIR_PATH)

    # Extract video info
    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO)
    video_info.width = int(video_info.width * SCALE_FACTOR)
    video_info.height = int(video_info.height * SCALE_FACTOR)

    # Initialize MaskAnnotator and BoxAnnotator
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.CLASS)
    box_annotator = sv.BoxAnnotator()


    # Define target video path
    TARGET_VIDEO = os.path.join(HOME, "video_data", f"{Path(SOURCE_VIDEO).stem}-result_sink.mp4")
    TARGET_VIDEO = Path(TARGET_VIDEO)
    os.makedirs(TARGET_VIDEO.parent, exist_ok=True)

    # Get sorted list of frame paths
    SOURCE_FRAME_PATHS = sorted(sv.list_files_with_extensions(SOURCE_FRAMES.as_posix(), extensions=["jpeg"]))

    # initialize the propagation index
    propagation_index = 0

    object_masks = {}  # Store object_id -> mask mapping

    # start the propagation
    for frame_idx, object_ids, mask_logits in sam2_model.propagate_in_video(inference_state):
        frame_idx = propagation_index+index
        print(f"Processing frame {frame_idx}, mask_logits shape: {mask_logits.shape}")

        if frame_idx >= len(SOURCE_FRAME_PATHS):
            print(f"⚠️ Frame index {frame_idx} out of range! Skipping...")
            continue

        frame_path = SOURCE_FRAME_PATHS[frame_idx]
        frame = cv2.imread(frame_path)

        if frame is None or frame.size == 0:
            print(f"⚠️ Skipping unreadable frame {frame_idx}!")
            continue

        if frame.shape[2] == 1:  # Convert grayscale to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        masks = (mask_logits > 0.0).cpu().numpy()
        if masks.ndim == 4:
            masks = np.squeeze(masks, axis=1)

        # here its starts -=-=-

        if np.any(masks):
            for obj_id, mask in zip(object_ids, masks):
                object_masks[obj_id] = mask  # Store updated masks

        # Restore previously stored masks if any objects reappear
        restored_masks = []
        restored_object_ids = []

        for obj_id, mask in object_masks.items():
            if obj_id in object_ids or FLAGG:  # Ensure masks persist when FLAGG is true
                restored_masks.append(mask)
                restored_object_ids.append(obj_id)

        if restored_masks:
            masks = np.array(restored_masks)
            object_ids = np.array(restored_object_ids)

        #here it ends -=-=-

        if not np.any(masks):
            print(f"Skipping segmentation for frame {frame_idx}: No segmentation found")
            cv2.imshow("Frame", frame)
            cv2.waitKey(1)
            cv2.imwrite(f"segmented_frames/{TARGET_VIDEO.stem}_frame_{frame_idx}_original.jpg", frame)
            continue

        # convert masks to BB
        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks=masks),
            mask=masks,
            class_id=np.array(object_ids)
        )

        # annotate box on the image
        # annotated_frame = mask_annotator.annotate(scene=frame.copy(), detections=detections)
        if toggle_flag:
            annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections, labels=labels)
        else:
            annotated_frame = mask_annotator.annotate(scene=frame.copy(), detections=detections)


        # condition to check empty frames
        if annotated_frame is None or annotated_frame.size == 0:
            print(f"Skipping empty frame {frame_idx}!")
            cv2.imshow("Frame", frame)
            cv2.waitKey(1)
            cv2.imwrite(f"segmented_frames/{TARGET_VIDEO.stem}_frame_{frame_idx}_original.jpg", frame)
            # cv2.imwrite(f"segmented_frames/frame_{frame_idx}.jpg", frame)

            continue

        print(f"Savingg segmented frame {frame_idx}...")
        print('imshow')
        #cv2.imshow does not work in new thread
        # cv2.imshow("Segmented Frame", annotated_frame)
        print('before waitkey')
        cv2.waitKey(1)
        print('after waitkey')
        # DISPLAY_PATH = f"segmented_frames/frame_{frame_idx}.jpg"        
        cv2.imwrite(f"segmented_frames/frame_{frame_idx}.jpg", annotated_frame)
        cv2.imwrite(DISPLAY_PATH, annotated_frame)

        # update propagation index
        propagation_index+=1


    # with open(interrupt_file, "r") as f:
    #     interrupt = f.read().strip()
    print('check for file exists------------')
    if os.path.exists(interrupt_file):
        with open(interrupt_file, "r") as f:
            interrupt = f.read().strip()
        if interrupt=="Resumed":
            print('file found resumed FLAGG changed -------====-=-=-=-=-')
            FLAGG = (mask_logits > 0.0).cpu().numpy()
    

    cv2.destroyAllWindows()
    return video_info 
