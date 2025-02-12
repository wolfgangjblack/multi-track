import os
import cv2

def extract_frames(video_path, output_dir, fps=10):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Get video properties
    original_fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(original_fps / fps)
    
    # Initialize frame counter
    frame_count = 0
    saved_count = 0
    
    while True:
        success, frame = video.read()
        
        if not success:
            break
            
        # Save frame at desired intervals
        if frame_count % frame_interval == 0:
            frame_name = f'frame_{saved_count:04d}.jpg'
            output_path = os.path.join(output_dir, frame_name)
            cv2.imwrite(output_path, frame)
            saved_count += 1
            
        frame_count += 1
    
    # Clean up
    video.release()
    print(f'Extracted {saved_count} frames at {fps} FPS')
    print(f'Frames saved to: {output_dir}')