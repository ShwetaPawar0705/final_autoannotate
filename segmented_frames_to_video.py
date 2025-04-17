import os
import cv2

#function to dynamically create a video from frames
def frames_to_video(input_frames_dir, output_video_path, frame_rate=30, width=1920, height=1080):

    # Create a VideoWriter object to write frames to a video file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    # Iterate through the sorted frame files
    for index in range(0, len(os.listdir(input_frames_dir))):
        frame_file = 'frame_'+str(index)+'.jpg'
        frame_path = os.path.join(input_frames_dir, frame_file)
        print(frame_path)
        
        # Make sure the file is an image
        if frame_file.lower().endswith(('jpeg', 'jpg', 'png')):
            frame = cv2.imread(frame_path)  # Read the image
            resized_frame = cv2.resize(frame, (width, height))  # Resize if needed
            video_writer.write(resized_frame)  # Write the frame to the video

    video_writer.release()  # Release the VideoWriter object



# Example usage
# input_frames_dir = 'segmented_frames'  # Directory containing frames
# output_video_path = f'video_data/basketball_1_result_FINAL.mp4'  # Path for the output video
# frames_to_video(input_frames_dir, output_video_path, frame_rate=video_info.fps, width=video_info.width, height=video_info.height)

