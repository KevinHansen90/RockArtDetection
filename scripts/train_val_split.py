import os
import random
from shutil import copy2
from collections import defaultdict


def load_class_names(file_path):
    with open(file_path, 'r') as f:
        class_names = f.read().splitlines()
    return class_names


def count_labels(label_files, label_dir, label_mapping):
    original_counts = {}
    mapped_counts = {}

    for label_file in label_files:
        label_path = os.path.join(label_dir, label_file)
        with open(label_path, 'r') as lf:
            lines = lf.readlines()
            for line in lines:
                original_label = line.strip().split()[0]
                mapped_label = label_mapping.get(original_label, original_label)

                # Count original labels
                if original_label in original_counts:
                    original_counts[original_label] += 1
                else:
                    original_counts[original_label] = 1

                # Count mapped labels
                if mapped_label in mapped_counts:
                    mapped_counts[mapped_label] += 1
                else:
                    mapped_counts[mapped_label] = 1

    return original_counts, mapped_counts


def train_val_split(image_dir, label_dir, train_size, val_size, target_labels, label_mapping):
    # Get all files from the directory
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.JPG'))])
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])

    # Ensure there's no mismatch between image and label files
    assert len(image_files) == len(label_files), "Mismatch between image and label files"

    # Load original and mapped class names
    original_class_names = load_class_names("/Users/kevinhansen/Documents/Git/RockArtDetection/data/labels.txt")
    mapped_class_names = load_class_names("/Users/kevinhansen/Documents/Git/RockArtDetection/data/labels_grouped.txt")

    # Count labels before processing
    original_counts, mapped_counts = count_labels(label_files, label_dir, label_mapping)

    # Print original class counts with names
    print("Original class counts before filtering:")
    for label, count in original_counts.items():
        class_name = original_class_names[int(label)]
        print(f"{label}: {class_name} - {count}")

    # Print mapped class counts with names
    print("Mapped class counts before filtering:")
    for label, count in mapped_counts.items():
        class_name = mapped_class_names[int(label)]
        print(f"{label}: {class_name} - {count}")

    # Pair image and label files
    paired_files = list(zip(image_files, label_files))

    # Filter and balance pairs based on the target labels
    if target_labels is not None:
        paired_files = filter_and_balance_pairs(paired_files, label_dir, target_labels, label_mapping,
                                                train_size + val_size)

    random.shuffle(paired_files)

    train_files = paired_files[:train_size]
    val_files = paired_files[train_size:train_size + val_size]

    # Define directories for train/val splits
    train_image_dir = "/data/train/images"
    train_label_dir = "/data/train/labels"
    val_image_dir = "/data/val/images"
    val_label_dir = "/data/val/labels"

    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_image_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)

    # Process and copy files
    process_and_copy(train_files, image_dir, label_dir, train_image_dir, train_label_dir, target_labels, label_mapping)
    process_and_copy(val_files, image_dir, label_dir, val_image_dir, val_label_dir, target_labels, label_mapping)


def filter_pairs_by_label(paired_files, label_dir, target_label, label_mapping):
    filtered_pairs = []
    for image_file, label_file in paired_files:
        label_path = os.path.join(label_dir, label_file)
        with open(label_path, 'r') as lf:
            lines = lf.readlines()
            for line in lines:
                original_label = line.strip().split()[0]
                mapped_label = label_mapping.get(original_label, original_label)

                if mapped_label == str(target_label):
                    filtered_pairs.append((image_file, label_file))
                    break  # Only need one matching label to include the image

        if len(filtered_pairs) >= train_size + val_size:
            break  # Stop if we have enough pairs

    if len(filtered_pairs) < train_size + val_size:
        raise ValueError(
            f"Not enough images with the target label {target_label} to meet the desired train and val sizes.")

    return filtered_pairs


def filter_and_balance_pairs(paired_files, label_dir, target_labels, label_mapping, total_size):
    class_pairs = defaultdict(list)

    for image_file, label_file in paired_files:
        label_path = os.path.join(label_dir, label_file)
        with open(label_path, 'r') as lf:
            lines = lf.readlines()
            for line in lines:
                original_label = line.strip().split()[0]
                mapped_label = label_mapping.get(original_label, original_label)

                if mapped_label in map(str, target_labels):
                    class_pairs[mapped_label].append((image_file, label_file))
                    break  # Only need one matching label to include the image

    # Balance the classes
    min_class_size = min(len(pairs) for pairs in class_pairs.values())
    balanced_size = min(min_class_size, total_size // len(target_labels))

    balanced_pairs = []
    for label in map(str, target_labels):
        balanced_pairs.extend(random.sample(class_pairs[label], balanced_size))

    # If we need more samples to reach the total_size, add them randomly
    if len(balanced_pairs) < total_size:
        remaining_pairs = [pair for pairs in class_pairs.values() for pair in pairs if pair not in balanced_pairs]
        balanced_pairs.extend(
            random.sample(remaining_pairs, min(len(remaining_pairs), total_size - len(balanced_pairs))))

    return balanced_pairs


def process_and_copy(files, image_dir, label_dir, target_image_dir, target_label_dir, target_labels, label_mapping):
    for image_file, label_file in files:
        label_path = os.path.join(label_dir, label_file)
        with open(label_path, 'r') as lf:
            lines = lf.readlines()

        with open(os.path.join(target_label_dir, label_file), 'w') as target_lf:
            for line in lines:
                parts = line.strip().split()
                original_label = parts[0]
                mapped_label = label_mapping.get(original_label, original_label)

                if target_labels is None or mapped_label in map(str, target_labels):
                    # Write the line to the new label file
                    target_lf.write(' '.join([mapped_label] + parts[1:]) + '\n')

        # Copy the corresponding image file only if it's not the same as the destination
        src_image_path = os.path.join(image_dir, image_file)
        dst_image_path = os.path.join(target_image_dir, image_file)

        if src_image_path != dst_image_path:
            copy2(src_image_path, dst_image_path)


# Example usage
label_mapping = {
    "0": "1",  # Animal
    "1": "1",
    "2": "1",
    "3": "1",
    "4": "3",  # Human
    "5": "2",  # Hand
    "6": "2",
    "7": "4",  # Animal_print
    "8": "2",  # Hand
    "9": "5",  # Geometric
    "10": "5",
    "11": "5",
    "12": "5",
    "13": "6",  # Other
    "14": "6",
    "15": "5",  # Geometric
    "16": "5",
    "17": "5",
    "18": "4"  # Animal_print
}

image_dir = "/Users/kevinhansen/Documents/Git/RockArtDetection/data/processed/images"
label_dir = "/Users/kevinhansen/Documents/Git/RockArtDetection/data/processed/labels"
train_size = 128
val_size = 32
target_labels = [1]

train_val_split(image_dir, label_dir, train_size, val_size, target_labels, label_mapping)
