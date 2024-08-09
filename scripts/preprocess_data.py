import os
import sys

# Adjust the Python path to include the parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.preprocessing import process_and_save_images


def main():
	# Adjust the paths to go one level up from the scripts directory
	script_dir = os.path.dirname(__file__)
	input_image_dir = os.path.join(script_dir, '..', 'data', 'raw')
	input_annotation_dir = os.path.join(script_dir, '..', 'data', 'annotations')
	output_image_dir = os.path.join(script_dir, '..', 'data', 'processed')
	output_annotation_dir = os.path.join(script_dir, '..', 'data', 'annotations_processed')
	target_size = 512  # or any size you find appropriate based on research

	# Print the absolute paths for debugging
	print(f"Input image directory: {os.path.abspath(input_image_dir)}")
	print(f"Input annotation directory: {os.path.abspath(input_annotation_dir)}")
	print(f"Output image directory: {os.path.abspath(output_image_dir)}")
	print(f"Output annotation directory: {os.path.abspath(output_annotation_dir)}")

	process_and_save_images(input_image_dir, input_annotation_dir, output_image_dir, output_annotation_dir, target_size)


if __name__ == "__main__":
	main()
