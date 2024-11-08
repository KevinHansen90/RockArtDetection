import os
import cv2
import numpy as np

def get_class_colors():
    """Define a fixed color for each specific class."""
    class_colors = {
        "Zoomorfo (artiodactyla)": (255, 0, 0),      # Red
        "Zoomorfo (ave)": (0, 255, 0),               # Green
        "Zoomorfo (piche)": (0, 0, 255),             # Blue
        "Zoomorfo (matuasto)": (255, 255, 0),        # Cyan
        "Antropomorfo": (255, 0, 255),               # Magenta
        "Positivo de mano": (0, 255, 255),           # Yellow
        "Negativo de mano": (128, 0, 0),             # Maroon
        "Negativo de pata de choique": (128, 128, 0),# Olive
        "Negativo de puño": (0, 128, 0),             # Dark Green
        "Circulos": (128, 0, 128),                   # Purple
        "Circulos concéntricos": (0, 128, 128),      # Teal
        "Lineas rectas": (0, 0, 128),                # Navy
        "Lineas zigzag": (128, 128, 128),            # Gray
        "Escala": (64, 0, 128),                      # Dark Purple
        "Persona": (0, 64, 128),                     # Dark Blue
        "Lazo bola": (64, 128, 64),                  # Dark Green
        "Conjuntos de puntos": (128, 64, 0),         # Brown
        "Impactos": (0, 128, 64),                    # Sea Green
        "Tridigitos": (128, 64, 128)                 # Light Purple
    }
    return class_colors

def draw_labels(image_path, yolo_txt_path, label_names_path, output_dir):
    # Read the image
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    # Load the label names
    with open(label_names_path, 'r') as f:
        label_names = f.read().splitlines()

    # Get the color for each specific label
    class_colors = get_class_colors()

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

        # Select color based on class name
        label_name = label_names[int(class_id)]
        color = class_colors.get(label_name, (255, 255, 255))  # Default to white if class not found

        # Draw the bounding box with a thicker border
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)  # Thicker border (4 pixels)

    # # Create a color legend below the image
    # legend_height = 30 * len(class_colors)  # Increased height for better spacing
    # legend = np.ones((legend_height, image.shape[1], 3), dtype=np.uint8) * 255
    #
    # y_offset = 25
    # for i, (class_name, color) in enumerate(class_colors.items()):
    #     x_offset = 10 if i % 2 == 0 else (image.shape[1] // 2) + 10  # Adjust x position for left and right columns
    #     cv2.rectangle(legend, (x_offset - 5, y_offset - 15), (x_offset + 20, y_offset), color, -1)  # Larger color box
    #     cv2.putText(legend, class_name, (x_offset + 30, y_offset - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    #     if i % 2 != 0:
    #         y_offset += 35  # Adjust line spacing
    #
    # # Combine the image and legend
    # output_image = np.vstack((image, legend))

    output_image = image

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the labeled image
    image_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, image_name)
    cv2.imwrite(output_path, output_image)

    print(f"Labeled image saved to {output_path}")


# Example usage:
image_path = '/Users/kevinhansen/Documents/Git/RockArtDetection/data/processed/images/DSC_0227_18.jpg'
yolo_txt_path = '/Users/kevinhansen/Documents/Git/RockArtDetection/data/processed/labels/DSC_0227_18.txt'
label_names_path = '/Users/kevinhansen/Documents/Git/RockArtDetection/data/labels.txt'
output_dir = '/Users/kevinhansen/Documents/Git/RockArtDetection/data'

draw_labels(image_path, yolo_txt_path, label_names_path, output_dir)


