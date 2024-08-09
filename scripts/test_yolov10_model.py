import os
from ultralytics import YOLOv10
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Define the function to load and preprocess the image
def load_image(image_path):
	return Image.open(image_path)


# Function to visualize the results
def visualize_results(image_path, predictions, output_path):
	image = Image.open(image_path)
	plt.figure(figsize=(10, 10))
	plt.imshow(image)
	ax = plt.gca()

	for prediction in predictions:
		box = prediction['box']
		score = prediction['confidence']
		label = prediction['class']

		x_min, y_min, x_max, y_max = box
		rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r',
								 facecolor='none')
		ax.add_patch(rect)
		ax.text(x_min, y_min, f'{label}: {score:.2f}', bbox=dict(facecolor='yellow', alpha=0.5))

	plt.axis('off')
	plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
	plt.close()


# Load the pre-trained YOLOv10 model
model = YOLOv10.from_pretrained('jameslahm/yolov10x')

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

		# Run inference
		results = model.predict(source=image_path, save=False)

		# Process the results
		predictions = []
		for result in results:
			for pred in result.boxes.data:
				box = pred[:4].cpu().numpy().tolist()
				score = pred[4].item()
				label = int(pred[5].item())
				predictions.append({
					'box': box,
					'confidence': score,
					'class': model.names[label]  # Convert label index to class name
				})

		# Visualize and save the results
		output_path = os.path.join(output_dir, f"result_{idx + 1}.png")
		visualize_results(image_path, predictions, output_path)
		print(f"Saved result to {output_path}")

print("Inference complete.")
