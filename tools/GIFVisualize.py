import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageSequence

class GIFVisualize:
    def __init__(self):
        pass

    def extract_multiple_gif(self, gif_path, output_path, descriptions):

        # Load the GIF
        gif = Image.open(gif_path)

        # Extract frames
        frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]

        # Select specific frames (0, 3, 6, 9)
        selected_indices = [0, 3, 6, 9]
        selected_frames = [frames[i] for i in selected_indices]

        # Display selected frames vertically with a gap between each row
        fig, axes = plt.subplots(len(selected_frames) + 1, 1, figsize=(10, len(selected_frames) * 5))
        axes[0].axis('off')

        # Add text descriptions above the first frame
        for i, (desc1, desc2) in enumerate(descriptions):
            x_position = (i + 3) / 9
            axes[0].text(x_position, 0.35, desc1, ha='center', va='center', fontsize=7)
            axes[0].text(x_position, 0.2, desc2, ha='center', va='center', fontsize=7)

        for ax, frame, i in zip(axes[1:], selected_frames, selected_indices):
            ax.imshow(frame)
            ax.axis('off')
            ax.set_frame_on(False)
            ax.text(0.02, 0.95, f"Frame {i}", color='gray', fontsize=6, 
                    ha='left', va='top', transform=ax.transAxes)

        plt.subplots_adjust(hspace=0.5)  # Adjust the gap between rows
        plt.tight_layout()

        # Save the result with higher DPI
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.show()

    def extract_single_gif(self, gif_path, output_path):
        gif = Image.open(gif_path)

        # Extract frames
        frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]

        # Select specific frames (0, 3, 6, 9)
        selected_indices = [0, 3, 6, 9]
        selected_frames = [frames[i] for i in selected_indices]

        # Display selected frames vertically with a gap between each row
        fig, axes = plt.subplots(len(selected_frames) + 1, 1, figsize=(10, len(selected_frames) * 5))
        axes[0].axis('off')

        for ax, frame, i in zip(axes[1:], selected_frames, selected_indices):
            ax.imshow(frame)
            ax.axis('off')
            ax.set_frame_on(False)
            ax.text(0.02, 0.95, f"Frame {i}", color='gray', fontsize=6, 
                    ha='left', va='top', transform=ax.transAxes)

        plt.subplots_adjust(hspace=0.5)  # Adjust the gap between rows
        plt.tight_layout()

        # Save the result with higher DPI
        plt.savefig("output_path", bbox_inches='tight', dpi=300)
        plt.show()
    
    def extract_single_video(self, video_path, output_path, descriptions):
        video = cv2.VideoCapture(video_path)

        # Extract frames
        frames = []
        success, frame = video.read()
        while success:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert to RGB
            success, frame = video.read()
        video.release()

        # Select specific frames (0, 3, 6, 9)
        selected_indices = [0, 7, 14, 21]
        selected_frames = [frames[i] for i in selected_indices]

        # Text descriptions for each column
        '''
        descriptions = [
            ["a man is walking", "with an umbrella"],
        ]
        '''
        # Display selected frames vertically with a gap between each row
        fig, axes = plt.subplots(len(selected_frames) + 1, 1, figsize=(10, len(selected_frames) * 5))
        axes[0].axis('off')

        # Add text descriptions above the first frame
        for i, (desc1, desc2) in enumerate(descriptions):
            x_position = (i + 4.5) / 9
            axes[0].text(x_position, 0.35, desc1, ha='center', va='center', fontsize=7)
            axes[0].text(x_position, 0.2, desc2, ha='center', va='center', fontsize=7)

        for ax, frame, i in zip(axes[1:], selected_frames, selected_indices):
            ax.imshow(frame)
            ax.axis('off')
            ax.set_frame_on(False)
            ax.text(0.02, 0.95, f"Frame {i}", color='gray', fontsize=6, 
                    ha='left', va='top', transform=ax.transAxes)

        plt.subplots_adjust(hspace=0.5)  # Adjust the gap between rows
        plt.tight_layout()

        # Save the result with higher DPI
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.show()

    description_dict = {

        "train_racket": [
            ["wonder woman is swinging", " a tennis racket"],
            ["a man is swinging", " a red tennis racket"],
            ["a man is swinging", " a silver tennis racket"],
            ["a man is swinging", " a wooden tennis racket"]
        ],

        "train_stool": [
            ["superman is sitting", " on a stool"],
            ["a man is sitting", " on a red stool"],
            ["a man is sitting", " on a metal stool"],
            ["a man is sitting", " on a wooden stool"]
        ],

        "train_suitcase": [
            ["batman is dragging", " a suitcase"],
            ["a man is dragging", " a red suitcase"],
            ["a man is dragging", " an aluminum suitcase"],
            ["a man is dragging", " an antiqued wooden suitcase"],
        ],

        "train_toolbox": [
            ["spiderman is carrying", " a toolbox"],
            ["a man is carrying", " a red toolbox"],
            ["a man is carrying", " a metal toolbox"],
            ["a man is carrying", " a wooden toolbox"],
        ],

        "valid_panda_dragging": [
            ["a panda dragging", "an orange suitcase"],
            ["a man is dragging", "an orange suitcase"],
            ["a fashionable man is", "dragging a black suitcase"],
            ["Barbie is dragging", "a suitcase"]
        ],
        "valid_skateboard_riding": [
            ["pikachu is riding", "a skateboard"],
            ["a woman is riding", "on a blue skateboard"],
            ["a woman is riding", "a snowboard"],
            ["a young boy, wearing a cap,", "is riding a skateboard"]
        ],
        "valid_racket_swinging": [
            ["mickey mouse is swinging", "a tennis racket"],
            ["a man is swinging", "a badminton racket"],
            ["a man is swinging", "a table tennis racket"],
            ["wonder woman is swinging", "a badminton racket"]
        ],
        "valid_umbrella_walking": [
            ["spider man is walking", "with an umbrella"],
            ["a man is walking", "with a red umbrella"],
            ["a man is walking", "with a metallic umbrella"],
            ["a man is walking", "with a wooden umbrella"]
        ],
        "valid_toolbox_carrying": [
            ["Homer Simpson is carrying", "a toolbox"],
            ["a woman is carrying", "a red toolbox"],
            ["a boy is carrying", "a toy toolbox"],
            ["super mario is carrying", "a toolbox"]
        ]
    }

    def get_description(self, category):
        # Accessing the class attribute within a method
        return self.description_dict.get(category, "Category not found")
    
    description_single_dict = {

        "racket": [
            ["a man is swinging", " a tennis racket"],
        ],

        "stool": [
            ["a man is sitting", " on a stool"],
        ],

        "suitcase": [
            ["a man is dragging", " a suitcase"],
        ],

        "toolbox": [
            ["a man is carrying", " a toolbox"],
        ],

    }

    def get_single_description(self, category):
        # Accessing the class attribute within a method
        return self.description_single_dict.get(category, "Category not found")
    
    # Example usage
if __name__ == "__main__":
    gif_visualizer = GIFVisualize()
    '''
    object_str = "stool"
    gif_path = "./my_data/output/pretending/" + object_str + ".gif"
    output_path = "./my_data/visualization/pretending/" + object_str + ".png"
    descriptions = gif_visualizer.get_description("train_" + object_str)
    gif_visualizer.extract_multiple_gif(gif_path, output_path, descriptions)
    '''
    object_str = "toolbox"
    video_path = "./my_data/input/pretending/" + object_str + ".mp4"
    output_path = "./my_data/visualization/input_visualize/pretending/" + object_str + ".png"
    description = gif_visualizer.get_single_description(object_str)
    gif_visualizer.extract_single_video(video_path, output_path, description)