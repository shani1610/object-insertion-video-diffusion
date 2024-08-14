import cv2
import moviepy.editor as mp

class VideoTools:
    def __init__(self):
        pass

    def resize_video(self, input_path, output_path, size=(512, 512)):
        # Open the input video
        cap = cv2.VideoCapture(input_path)

        # Get video properties
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Ensure the input video is square
        if frame_width != frame_height:
            raise ValueError("Input video frames must be square.")

        # Open the output video writer
        out = cv2.VideoWriter(output_path, fourcc, fps, size)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame
            resized_frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

            # Write the frame to the output video
            out.write(resized_frame)

        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    # You can add more methods for other video processing tasks

    def convert_mov_to_mp4(self, input_path, output_path):
        clip = mp.VideoFileClip(input_path)
        clip.write_videofile(output_path)

    def trim_video(self, video_path, start_time_seconds=0, frames_to_capture=24, fps=30):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Cannot open video file.")
            return None
        
        # Calculate the start frame based on the start time and fps
        start_frame = int(start_time_seconds * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Define the codec and create VideoWriter object
        output_path_trimmed = "video_trimmed.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_trimmed = cv2.VideoWriter(output_path_trimmed, fourcc, fps, (frame_width, frame_height))
        
        frame_counter = 0
        while frame_counter < frames_to_capture:
            ret, frame = cap.read()
            if not ret:
                break
            out_trimmed.write(frame)
            frame_counter += 1
        
        cap.release()
        out_trimmed.release()
        print(f"Video trimmed and saved to {output_path_trimmed}")

    def extract_frames(self, video_path, output_folder):
        cap = cv2.VideoCapture(video_path)
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(f"{output_folder}/frame_{count:04d}.png", frame)
            count += 1
        cap.release()
        cv2.destroyAllWindows()

    def extract_first_frame(self, video_path, output_folder):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cv2.imwrite(f"{output_folder}/frame_03.png", frame)
        cap.release()
        cv2.destroyAllWindows()

    def count_frames(self, video_path):
        # Open the video file
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
        else:
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
            cap.release()

        print(f"Total frames counted: {frame_count}")

# Example usage
if __name__ == "__main__":
    video_tools = VideoTools()

    
    # # trim video example
    # input_video_path = './videos/my_data/IMG_6655.mov'
    # output_video_path = './videos/my_data/IMG_66555.mp4'

    # video_tools.convert_mov_to_mp4(input_video_path, output_video_path)

    # # resize video example
    # input_video_path = output_video_path
    # video_tools.resize_video(input_video_path, './videos/my_data/stool_prend.mp4')

    # video_tools.count_frames('./videos/my_data/stool_prend.mp4')
    

    # video_tools.trim_video('./videos/my_data/stool_prend.mp4', start_time_seconds=0, frames_to_capture=24, fps=30)


    #video_tools.extract_first_frame("./videos/umbrella.mp4", "./videos/stuff/")
    video_tools.extract_first_frame("./videos/inpainted/umbrella.mp4", "./videos/stuff/")
