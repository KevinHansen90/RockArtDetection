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


# Function to clamp a value within the min and max bounds
def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


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
                    tile_width_actual = box[2] - box[0]
                    tile_height_actual = box[3] - box[1]

                    # Calculate the new bounding boxes for this tile
                    new_annotations = []
                    for line in annotations:
                        class_id, x_center, y_center, width, height = map(float, line.split())

                        # Convert to absolute coordinates
                        abs_x_center = x_center * image_width
                        abs_y_center = y_center * image_height
                        abs_width = width * image_width
                        abs_height = height * image_height

                        # Calculate the intersection of the bounding box with the tile
                        x1 = max(x, abs_x_center - abs_width / 2)
                        y1 = max(y, abs_y_center - abs_height / 2)
                        x2 = min(x + tile_width_actual, abs_x_center + abs_width / 2)
                        y2 = min(y + tile_height_actual, abs_y_center + abs_height / 2)

                        # Check if there's an intersection
                        if x1 < x2 and y1 < y2:
                            # Calculate new relative coordinates
                            new_x_center = (x1 + x2) / 2 - x
                            new_y_center = (y1 + y2) / 2 - y
                            new_width = x2 - x1
                            new_height = y2 - y1

                            # Convert to YOLO format (relative to tile size)
                            new_x_center /= tile_width_actual
                            new_y_center /= tile_height_actual
                            new_width /= tile_width_actual
                            new_height /= tile_height_actual

                            new_annotations.append(
                                f"{int(class_id)} {new_x_center:.6f} {new_y_center:.6f} {new_width:.6f} {new_height:.6f}\n")

                    if new_annotations:
                        tile_idx += 1
                        tile_filename = os.path.splitext(filename)[0] + f"_{tile_idx}.jpg"
                        tile_annotation_filename = os.path.splitext(filename)[0] + f"_{tile_idx}.txt"

                        # Save the tile image with quality and aspect ratio preserved
                        tile.save(os.path.join(output_image_dir, tile_filename), "JPEG", quality=100)

                        # Save the new annotations
                        with open(os.path.join(output_annotation_dir, tile_annotation_filename), 'w') as ann_file:
                            ann_file.writelines(new_annotations)


# Define directories
image_dir = '/Users/kevinhansen/Documents/Git/RockArtDetection/data/raw/images'
annotation_dir = '/Users/kevinhansen/Documents/Git/RockArtDetection/data/raw/labels'
output_image_dir = '/Users/kevinhansen/Documents/Git/RockArtDetection/data/processed/images'
output_annotation_dir = '/Users/kevinhansen/Documents/Git/RockArtDetection/data/processed/labels'

# Run the script
divide_images(image_dir, annotation_dir, output_image_dir, output_annotation_dir)

