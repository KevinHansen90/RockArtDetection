import os
import torch
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define the function to load and preprocess the image
def load_image(image_path):
    return Image.open(image_path)

# Function to visualize the results
def visualize_results(image, results, output_path, id2label):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()

    for result in results:
        for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
            score, label = score.item(), label_id.item()
            box = [round(i, 2) for i in box.tolist()]
            x_min, y_min, x_max, y_max = box
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            label_name = id2label[label]
            ax.text(x_min, y_min, f'{label_name}: {score:.2f}', bbox=dict(facecolor='yellow', alpha=0.5))

    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Load the pre-trained model and processor
image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd")

# Get the label map
id2label = model.config.id2label

# Directories
image_dir = '../data/raw'
output_dir = '../output/test_results'

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Run inference on the dataset
for idx, filename in enumerate(os.listdir(image_dir)):
    if filename.endswith((".jpg", ".jpeg", ".png", ".JPG")):
        image_path = os.path.join(image_dir, filename)
        image = load_image(image_path)

        # Preprocess the image
        inputs = image_processor(images=image, return_tensors="pt")

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process the results
        results = image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([image.size[::-1]]), threshold=0.3)

        # Visualize and save the results
        output_path = os.path.join(output_dir, f"result_{idx + 1}.png")
        visualize_results(image, results, output_path, id2label)
        print(f"Saved result to {output_path}")

print("Inference complete.")


