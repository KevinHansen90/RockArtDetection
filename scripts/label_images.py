import os
import cv2


def get_color_palette(num_classes):
    """Define a fixed color palette."""
    palette = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 0),  # Maroon
        (128, 128, 0),  # Olive
        (0, 128, 0),  # Dark Green
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
        (0, 0, 128),  # Navy
    ]

    # If the number of classes exceeds the palette, repeat colors
    if num_classes > len(palette):
        palette *= (num_classes // len(palette)) + 1

    return palette[:num_classes]


def draw_labels(image_path, yolo_txt_path, label_names_path, output_dir):
    # Read the image
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    # Load the label names
    with open(label_names_path, 'r') as f:
        label_names = f.read().splitlines()

    # Get a fixed color palette for the labels
    colors = get_color_palette(len(label_names))

    # Load YOLO labels
    with open(yolo_txt_path, 'r') as f:
        lines = f.read().splitlines()

    # Draw the labels on the image
    for line in lines:
        class_id, x_center, y_center, width, height = map(float, line.split())
        x_center, y_center, width, height = int(x_center * w), int(y_center * h), int(width * w), int(height * h)

        # Calculate bounding box coordinates
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # Select color based on class ID
        color = colors[int(class_id)]

        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Draw the label name
        label = label_names[int(class_id)]
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the labeled image
    image_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, image_name)
    cv2.imwrite(output_path, image)

    print(f"Labeled image saved to {output_path}")

# Example usage:
image_path = '/Users/kevinhansen/Documents/Git/RockArtDetection/data/raw/DSC_0361.jpg'
yolo_txt_path = '/Users/kevinhansen/Documents/Git/RockArtDetection/data/annotations/DSC_0361.txt'
label_names_path = '/Users/kevinhansen/Documents/Git/RockArtDetection/data/labels.txt'
output_dir = '/Users/kevinhansen/Documents/Git/RockArtDetection/data'

draw_labels(image_path, yolo_txt_path, label_names_path, output_dir)
