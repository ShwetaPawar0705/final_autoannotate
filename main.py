
import os
import cv2
import shutil
import supervision as sv
from path_config import HOME
from SAM import segment_instances   # Function to create 1st segmentation for every skip frame
# from V_segment_masks import plot_and_save_masks       # Plotting objects masks in folder -> segment_masks
from mask_auto_annotation import auto_BB_annotate     # Function for bounding box annotation
from segmented_frames_to_video import frames_to_video       # Create video from segmented frames(final)
from SAM_for_video import bbox_to_sam, visualize_segment_video, video_to_frames, FRAMES_DIR_PATH     # For video_propagation 
from Augmentation import augment_and_zip
#
#
#
# from fasttpi import tasks, FILE_ID  #to check if the dictionary if updated for interrupt


# this is the main function to execute the entire flow of the project
def main_func(input_path, output_path, input_classes, file_id,desired_fps,confidence_threshold ,toggle_flag):

    ZIP_OUTPUT_DIR = "segmented_frames_annotated"    # temporary location to store zips
    SCALE_FACTOR = 1.0

    # Extract video info
    video_info = sv.VideoInfo.from_video_path(input_path)
    video_info.width = int(video_info.width * SCALE_FACTOR)
    video_info.height = int(video_info.height * SCALE_FACTOR)

    # video_info=[]
    # video_info = None  # Initialize video_info before the loop
    terminator=0

    # Define source directory for images
    SOURCE_DIR = f"{HOME}/data"
    OUTPUT_DIR_SM = f"{HOME}/segment_masks"     # for storing individual object masks
    NO_OF_OBJECTS = 'no_of_objects'
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR_SM, exist_ok=True)

    # Ensure that object count dir exists
    os.makedirs(NO_OF_OBJECTS, exist_ok=True)

    # objects classes to detect, passed by user
    CLASSES = input_classes

    
    # #1 generate frames from video 
    # fps, totalNoFrames=video_to_frames(input_path, FRAMES_DIR_PATH)
    # print(f'\nfps: {fps}\ntotal number of frames: {totalNoFrames}\n\n')
    # skip_frames = int(fps/5)

    #1 generate frames from video 
    fps, totalNoFrames=video_to_frames(input_path, FRAMES_DIR_PATH)
    print(f'\nfps: {fps}\ntotal number of frames: {totalNoFrames}\n\n')
    final_skip_frames = int(totalNoFrames*(desired_fps/fps))
    skip_frames = int(totalNoFrames/final_skip_frames)
    print("skip_frames: ", final_skip_frames)
    print("Final_skipped_frames: ", skip_frames)

    # Get all image files in the data directory
    image_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(('.jpeg', '.jpg', '.png'))]
    print(f'image_files: {type(image_files)}, len{len(image_files)}')


    # initialize imp variables
    FRAME_IDX=0
    toggle=0
    Cust_Pt_Flag = False

    # create the progress dir to be accessed by STATUS endpoint
    progress_dir = "progress"
    os.makedirs(progress_dir, exist_ok=True)
    progress_file = os.path.join(progress_dir, f"progress_{file_id}.txt")

    INTERRUPT_DIR = "interrupts"
    interrupt_file = os.path.join(INTERRUPT_DIR, f"interrupt_{file_id}.txt")
    print('read interrupt')

    # Loop through each image and process it__annote and segment loop (skipping fps/4 frames)
    for index, image_file in enumerate(sorted(image_files)):
        print('in loop')
        # get enumerated image path
        source_image_path = os.path.join(SOURCE_DIR, image_file)

        # condition for skipping frames based on fps
        if index % skip_frames != 0:
            
            # toggle variable to decide if skipped frames should be saved ...otherwise will be accessed later by the inference state
            if toggle == 1:
                frame = cv2.imread(source_image_path)
                cv2.imwrite(f"segmented_frames/frame_{index}.jpg", frame)
            continue

        # initialize inference index
        inference_index = index%skip_frames
        
        print(f"Processing: {source_image_path}")  # Debugging output

        # with open(interrupt_file, "r") as f:
        #     interrupt = f.read().strip()
        # if interrupt=="STOPDETECTION":
        #     print("New object detection STOPPED by user.")
        #     # pass
        
        #2 Perform bounding box annotation
        image, detections, labels, OUTPUT_IMAGE_PATH = auto_BB_annotate(source_image_path, CLASSES,confidence_threshold)

        print(f'detections.mask: {detections.xyxy}')
        print(f'labels: {labels}, object_count: {len(labels)}')

        with open(os.path.join(NO_OF_OBJECTS, f"no_of_objects_{file_id}.txt"), "w") as o_file:
            o_file.write(str(len(labels))+'\n')

        # no mask or object detected in frame(every {int(fps/4)}th frame taken)
        if len(detections.xyxy)==0:
            print("No detections found.")
                
            # toggle variable updated
            toggle = 1
            frame = cv2.imread(source_image_path)
            cv2.imwrite(f"segmented_frames/frame_{index}.jpg", frame)
            continue

        # toggle variable updated
        toggle=0

        #3 Perform instance segmentation
        mask = segment_instances(OUTPUT_IMAGE_PATH, image, detections, CLASSES)

        #
        #
        #
        if os.path.exists(interrupt_file):
            with open(interrupt_file, "r") as f:
                interrupt = f.read().strip()
            if interrupt=="Interrupted":
                print("Processing interrupted by user.")
                Cust_Pt_Flag = True
            elif interrupt=="STOPPROCESS":
                break


        # if tasks[FILE_ID] == "Interrupted":
        #     print("Processing interrupted by user.")
        #     Cust_Pt_Flag = True

        #4 Add detected masks to the model and initialize the inference state
        object_ids, mask_logits, inference_state = bbox_to_sam(file_id,Cust_Pt_Flag, skip_frames, SOURCE_DIR, source_image_path, FRAME_IDX, inference_index, detections, CLASSES, image, mask)
        
        #
        #
        #
        if Cust_Pt_Flag == True:
            # Save interrupt resumed
            print('flag true')
            with open(interrupt_file, "r+") as f:
                print('not reading this line')
                interrupt = f.read().strip()
                f.seek(0)
                f.write(str('Resumed')+'\n')
                f.truncate()  # Remove any leftover content from the previous data
                print('after not reading this line')
                if interrupt=="STOPDETECTION":
                    f.seek(0)
                    f.write(str('STOPDETECTION')+'\n')
                    f.truncate()  # Remove any leftover content from the previous data
            Cust_Pt_Flag = False
            print('Cust_Pt_Flag set back to False')

        print(f'FRAME_IDX: {FRAME_IDX}')
        
        # update the frame ID
        FRAME_IDX=min(FRAME_IDX+1, len(image_files))

        # to check if inference is not None
        assert inference_state is not None, "Error: inference_state is None!"

        #5 Propagate masks in video...
        video_info = visualize_segment_video(index, input_path, inference_state, file_id, toggle_flag, labels)
        print('Video propagation complete!!!....')

        progress = (index / totalNoFrames) * 100

        # Save progress
        with open(progress_file, "w") as f:
            f.write(str(int(progress))+'\n')

        

        print(f"{file_id} - Progress: {int(progress)}%")

        with open(interrupt_file, "r") as f:
            interrupt = f.read().strip()
        if interrupt=="STOPDETECTION":
            print("New object detection STOPPED by user.")
            if terminator==1:
                print('terminator is 1')
                break
            terminator=1
            print('terminator set to 1')
            #++++++++++++++++++++++++++++++++++++

            # object_ids, mask_logits, inference_state = bbox_to_sam(file_id,Cust_Pt_Flag, skip_frames, SOURCE_DIR, source_image_path, FRAME_IDX, inference_index, detections, CLASSES, image, mask)
            # # to check if inference is not None
            # assert inference_state is not None, "Error: inference_state is None!"

            # #5 Propagate masks in video...
            # video_info = visualize_segment_video(index, input_path, inference_state, file_id)
            # print('direct break')
            # break
            # pass

        # Save segmentation masks
        # plot_and_save_masks(detections, CLASSES, OUTPUT_DIR_SM, image_file)

    # initialize dirs for final frame to video conversion
    input_frames_dir = 'segmented_frames'  # Directory containing frames
    output_video_path = output_path     # path for output video

    # if video_info is None:
        # video_info.fps=30
        # video_info.width=1920
        # video_info.height=1080
        # pass
    #     raise ValueError("Error: video_info was never assigned. Check if frames were processed correctly.")

    print('before video info')
    #6 Frames to video conversion
    frames_to_video(input_frames_dir, output_video_path, frame_rate=video_info.fps, width=video_info.width, height=video_info.height)
    # frames_to_video(input_frames_dir, output_video_path, video_info)
    print('after video info')

    # Augment and zip segmented images
    augment_and_zip('instance_segmentation', file_id)
    augment_and_zip('instance_segmentation', file_id,2)
    augment_and_zip('instance_segmentation', file_id,3)

    folder_path = os.path.join("/home/aditya/Desktop/DO_NOT_OPEN/Img_auto-annotation/FINAL_CODE_AUTO_ANOTE/segmented_frames")
    zip_path = os.path.join(ZIP_OUTPUT_DIR, f"{file_id}_processed.zip")
    shutil.make_archive(base_name=zip_path.replace(".zip", ""), format='zip', root_dir=folder_path)


    dir_to_clear = ['segmented_frames','data','bounding_box','bbox_output','instance_segmentation','inference_frames']
    # dir_to_clear = ['data','bounding_box','bbox_output','instance_segmentation','inference_frames']
    for dir_path in dir_to_clear:
        print('deleting - ', dir_path)
        shutil.rmtree(dir_path, ignore_errors=True)  # Deletes everything
        print('recreating - ', dir_path)
        os.makedirs(dir_path, exist_ok=True)  # Recreates the empty directory
        
    # Save interrupt
    with open(interrupt_file, "w") as f:
        f.write(str("Ended")+'\n')
        print('Interrupt file updated to "Ended"')
    print("Processing complete!")

