import os
from PIL import Image


# Define function to read YOLO annotations
def read_yolo_annotation(file_path):
	if os.path.exists(file_path):
		with open(file_path, 'r') as file:
			lines = file.readlines()
			if lines:
				return lines
	return None


# Function to divide images into smaller sub-images and generate corresponding annotations
def divide_images(image_dir, annotation_dir, output_image_dir, output_annotation_dir, tile_size=(512, 512)):
	if not os.path.exists(output_image_dir):
		os.makedirs(output_image_dir)
	if not os.path.exists(output_annotation_dir):
		os.makedirs(output_annotation_dir)

	for filename in os.listdir(image_dir):
		if filename.lower().endswith((".jpg", ".jpeg", ".png")):
			image_path = os.path.join(image_dir, filename)
			annotation_path = os.path.join(annotation_dir, os.path.splitext(filename)[0] + ".txt")

			image = Image.open(image_path)
			image_width, image_height = image.size
			tile_width, tile_height = tile_size

			# Read annotations
			annotations = read_yolo_annotation(annotation_path)
			if annotations is None:
				continue

			tile_idx = 0
			for y in range(0, image_height, tile_height):
				for x in range(0, image_width, tile_width):
					# Define the bounding box of the tile
					box = (x, y, min(x + tile_width, image_width), min(y + tile_height, image_height))
					tile = image.crop(box)

					# Calculate the new bounding boxes for this tile
					new_annotations = []
					for line in annotations:
						class_id, x_center, y_center, width, height = map(float, line.split())
						abs_x_center = x_center * image_width
						abs_y_center = y_center * image_height
						abs_width = width * image_width
						abs_height = height * image_height

						if (x <= abs_x_center <= x + tile_width) and (y <= abs_y_center <= y + tile_height):
							new_x_center = (abs_x_center - x) / tile_width
							new_y_center = (abs_y_center - y) / tile_height
							new_width = abs_width / tile_width
							new_height = abs_height / tile_height
							new_annotations.append(
								f"{int(class_id)} {new_x_center} {new_y_center} {new_width} {new_height}\n")

					if new_annotations:
						tile_idx += 1
						tile_filename = os.path.splitext(filename)[0] + f"_{tile_idx}.jpg"
						tile_annotation_filename = os.path.splitext(filename)[0] + f"_{tile_idx}.txt"

						# Save the tile image
						tile.save(os.path.join(output_image_dir, tile_filename))

						# Save the new annotations
						with open(os.path.join(output_annotation_dir, tile_annotation_filename), 'w') as ann_file:
							ann_file.writelines(new_annotations)


# Define directories
image_dir = '/Users/kevinhansen/Documents/Git/RockArtDetection/data/raw'
annotation_dir = '/Users/kevinhansen/Documents/Git/RockArtDetection/data/annotations'
output_image_dir = '/Users/kevinhansen/Documents/Git/RockArtDetection/data/processed'
output_annotation_dir = '/Users/kevinhansen/Documents/Git/RockArtDetection/data/annotations_processed'

# Run the script
divide_images(image_dir, annotation_dir, output_image_dir, output_annotation_dir)