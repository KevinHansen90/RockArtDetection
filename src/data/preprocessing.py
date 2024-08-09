import cv2
import numpy as np
import os


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
	lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
	l, a, b = cv2.split(lab)
	clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
	cl = clahe.apply(l)
	limg = cv2.merge((cl, a, b))
	final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
	return final


def bilateral_filtering(image, d=9, sigma_color=75, sigma_space=75):
	return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def canny_edge_detection(image, low_threshold=50, high_threshold=150):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray, low_threshold, high_threshold)
	return edges


def combine_edge_with_original(image, edge_map, alpha=0.9, beta=0.1):
	return cv2.addWeighted(image, alpha, cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR), beta, 0)


def enhance_color(image, alpha=1.2, beta=25):
	enhanced_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
	return enhanced_image


def unsharp_mask(image, sigma=1.0, strength=1.5):
	blurred = cv2.GaussianBlur(image, (0, 0), sigma)
	sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
	return sharpened


def create_color_mask(image, lower_bound, upper_bound):
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, lower_bound, upper_bound)
	return mask


def blend_with_color_mask(image, mask, alpha=0.5, beta=0.5):
	color_mask = cv2.bitwise_and(image, image, mask=mask)
	blended = cv2.addWeighted(image, alpha, color_mask, beta, 0)
	return blended


def adjust_brightness_contrast(image, brightness=0, contrast=30):
	img = cv2.addWeighted(image, 1 + (contrast / 127.0), image, 0, brightness - contrast)
	return img


def convert_mask_to_white(mask):
	white_mask = np.zeros_like(mask)
	white_mask[mask > 0] = 255
	return white_mask


def resize_image(image, target_size):
	h, w = image.shape[:2]
	scaling_factor = target_size / float(max(h, w))
	new_size = (int(w * scaling_factor), int(h * scaling_factor))
	resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
	return resized_image, scaling_factor


def adjust_annotations(annotations, scaling_factor):
	adjusted_annotations = []
	for annotation in annotations:
		adjusted_annotation = [annotation[0], annotation[1] * scaling_factor, annotation[2] * scaling_factor,
							   annotation[3] * scaling_factor, annotation[4] * scaling_factor]
		adjusted_annotations.append(adjusted_annotation)
	return adjusted_annotations


def adjust_dynamic_range(image, lower_percentile=2, upper_percentile=98):
	lower_bound = np.percentile(image, lower_percentile)
	upper_bound = np.percentile(image, upper_percentile)
	adjusted_image = np.clip(image, lower_bound, upper_bound)
	adjusted_image = ((adjusted_image - lower_bound) / (upper_bound - lower_bound) * 255).astype(np.uint8)
	return adjusted_image


def preprocess_image(image_path, annotation_path, target_size=512):
	image = cv2.imread(image_path)
	with open(annotation_path, 'r') as f:
		annotations = [list(map(float, line.strip().split())) for line in f]

	resized_image, scaling_factor = resize_image(image, target_size)
	adjusted_annotations = adjust_annotations(annotations, scaling_factor)

	clahe_image = apply_clahe(resized_image)
	filtered_image = bilateral_filtering(clahe_image)
	edge_map = canny_edge_detection(filtered_image, low_threshold=50, high_threshold=150)
	combined_image = combine_edge_with_original(filtered_image, edge_map, alpha=0.9, beta=0.1)
	sharpened_image = unsharp_mask(combined_image, sigma=1.0, strength=1.5)
	enhanced_image = enhance_color(sharpened_image, alpha=1.2, beta=25)
	dynamic_range_adjusted_image = adjust_dynamic_range(enhanced_image)

	# Color masking for common colors in rock art
	lower_bound_red = np.array([0, 70, 50])
	upper_bound_red = np.array([10, 255, 255])
	lower_bound_red_upper = np.array([170, 70, 50])
	upper_bound_red_upper = np.array([180, 255, 255])
	lower_bound_yellow = np.array([20, 100, 100])
	upper_bound_yellow = np.array([30, 255, 255])

	red_mask = create_color_mask(dynamic_range_adjusted_image, lower_bound_red, upper_bound_red) + \
			   create_color_mask(dynamic_range_adjusted_image, lower_bound_red_upper, upper_bound_red_upper)
	yellow_mask = create_color_mask(dynamic_range_adjusted_image, lower_bound_yellow, upper_bound_yellow)

	red_white_mask = convert_mask_to_white(red_mask)
	yellow_white_mask = convert_mask_to_white(yellow_mask)

	blended_image_red = blend_with_color_mask(dynamic_range_adjusted_image, red_white_mask, alpha=0.5, beta=0.5)
	blended_image_yellow = blend_with_color_mask(dynamic_range_adjusted_image, yellow_white_mask, alpha=0.5, beta=0.5)

	final_image = cv2.addWeighted(blended_image_red, 0.5, blended_image_yellow, 0.5, 0)
	final_image = adjust_brightness_contrast(final_image, brightness=30, contrast=40)

	return final_image, adjusted_annotations


def process_and_save_images(input_image_dir, input_annotation_dir, output_image_dir, output_annotation_dir,
							target_size=512):
	if not os.path.exists(output_image_dir):
		os.makedirs(output_image_dir)
	if not os.path.exists(output_annotation_dir):
		os.makedirs(output_annotation_dir)

	for filename in os.listdir(input_image_dir):
		if filename.endswith((".jpg", ".jpeg", ".png", ".JPG")):
			image_path = os.path.join(input_image_dir, filename)
			annotation_path = os.path.join(input_annotation_dir, filename.rsplit('.', 1)[0] + '.txt')

			print(f"Processing image: {image_path}")
			print(f"Using annotation: {annotation_path}")

			if not os.path.exists(annotation_path):
				print(f"Annotation file not found: {annotation_path}")
				continue

			processed_image, adjusted_annotations = preprocess_image(image_path, annotation_path, target_size)

			output_image_path = os.path.join(output_image_dir, filename)
			output_annotation_path = os.path.join(output_annotation_dir, filename.rsplit('.', 1)[0] + '.txt')

			cv2.imwrite(output_image_path, processed_image)
			with open(output_annotation_path, 'w') as f:
				for annotation in adjusted_annotations:
					f.write(' '.join(map(str, annotation)) + '\n')
			print(f"Saved processed image to: {output_image_path}")
			print(f"Saved adjusted annotations to: {output_annotation_path}")

	print("Processing complete.")
