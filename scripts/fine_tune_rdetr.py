# import os
# import torch
# from torch.utils.data import Dataset, DataLoader
# from transformers import DetrForObjectDetection, DetrImageProcessor
# from PIL import Image
#
# # Define simplified class labels
# class_mapping = {
#     "Zoomorfo (artiodactyla)": "Animal",
#     "Zoomorfo (ave)": "Animal",
#     "Zoomorfo (piche)": "Animal",
#     "Zoomorfo (matuasto)": "Animal",
#     "Antropomorfo": "Human",
#     "Positivo de mano": "Hand",
#     "Negativo de mano": "Hand",
#     "Negativo de pata de choique": "Animal_print",
#     "Negativo de puño": "Hand",
#     "Círculos": "Geometric",
#     "Círculos concéntricos": "Geometric",
#     "Líneas rectas": "Geometric",
#     "Líneas zigzag": "Geometric",
#     "Escala": "Other",
#     "Persona": "Human",
#     "Lazo bola": "Other",
#     "Conjuntos de puntos": "Geometric",
#     "Impactos": "Other",
#     "Tridígitos": "Animal_print"
# }
#
# # Create a mapping of simplified labels to numerical IDs
# id2label = {i: label for i, label in enumerate(set(class_mapping.values()))}
# label2id = {label: i for i, label in id2label.items()}
#
# # Custom dataset class
# class CustomDataset(Dataset):
#     def __init__(self, image_dir, annotation_dir, processor, target_size=(512, 512)):
#         self.image_dir = image_dir
#         self.annotation_dir = annotation_dir
#         self.processor = processor
#         self.target_size = target_size
#         self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
#
#         if not self.image_files:
#             raise ValueError(f"No image files found in {image_dir}")
#
#     def __len__(self):
#         return len(self.image_files)
#
#     def __getitem__(self, idx):
#         img_name = self.image_files[idx]
#         img_path = os.path.join(self.image_dir, img_name)
#         ann_path = os.path.join(self.annotation_dir, img_name.rsplit('.', 1)[0] + '.txt')
#
#         if not os.path.exists(img_path):
#             raise FileNotFoundError(f"Image file not found: {img_path}")
#         if not os.path.exists(ann_path):
#             raise FileNotFoundError(f"Annotation file not found: {ann_path}")
#
#         image = Image.open(img_path).convert("RGB")
#         image = image.resize(self.target_size)  # Resize image to the target size
#         w, h = image.size
#
#         # Read YOLO annotations and convert to COCO format
#         with open(ann_path, 'r') as f:
#             annotations = f.readlines()
#
#         coco_annotations = []
#
#         for ann in annotations:
#             class_id, x_center, y_center, width, height = map(float, ann.strip().split())
#             x_min = (x_center - width / 2) * w
#             y_min = (y_center - height / 2) * h
#             x_max = (x_center + width / 2) * w
#             y_max = (y_center + height / 2) * h
#
#             original_class = list(class_mapping.keys())[int(class_id)]
#             simplified_class = class_mapping[original_class]
#             category_id = label2id[simplified_class]
#
#             coco_annotations.append({
#                 "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
#                 "category_id": category_id,
#                 "area": (x_max - x_min) * (y_max - y_min),
#                 "iscrowd": 0
#             })
#
#         if not coco_annotations:
#             raise ValueError(f"No valid annotations found for image: {img_name}")
#
#         target = {
#             "image_id": idx,
#             "annotations": coco_annotations
#         }
#
#         # Use the processor to prepare the image and annotations
#         encoding = self.processor(images=image, annotations=target, return_tensors="pt")
#
#         # Remove batch dimension for each item in encoding
#         for k, v in encoding.items():
#             if isinstance(v, torch.Tensor):
#                 encoding[k] = v.squeeze(0)
#             elif isinstance(v, list):
#                 encoding[k] = v[0] if v else v  # Take the first item if it's a non-empty list
#
#         return encoding
#
# # Custom collate function
# def collate_fn(batch):
#     pixel_values = torch.stack([item["pixel_values"] for item in batch])
#     encoding = {
#         "pixel_values": pixel_values,
#         "labels": [{k: v.to(pixel_values.device) for k, v in item["labels"].items()} for item in batch]
#     }
#     return encoding
#
# # Set up the model and processor
# model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101",
#                                                num_labels=len(id2label),
#                                                ignore_mismatched_sizes=True, revision="no_timm")
# processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101", revision="no_timm",
#                                                size={"shortest_edge": 800, "longest_edge": 1333})
#
# # Update the model's config with new labels
# model.config.id2label = id2label
# model.config.label2id = label2id
#
# # Set directories
# image_dir = '../data/processed'
# annotations_dir = '../data/annotations_processed'
# output_dir = '../output/fine_tuned_model'
#
# # Set up the dataset and dataloader
# try:
#     train_dataset = CustomDataset(image_dir, annotations_dir, processor)
#     train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
#     print(f"Dataset initialized with {len(train_dataset)} images")
# except Exception as e:
#     print(f"Error initializing dataset: {str(e)}")
#     exit(1)
#
# # Set up the optimizer
# optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
#
# # Training loop
# num_epochs = 20
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
#
# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0
#     for batch in train_dataloader:
#         pixel_values = batch["pixel_values"].to(device)
#         labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
#
#         outputs = model(pixel_values=pixel_values, labels=labels)
#
#         loss = outputs.loss
#         total_loss += loss.item()
#
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#
#     avg_loss = total_loss / len(train_dataloader)
#     print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
#
# # Save the fine-tuned model
# model.save_pretrained(output_dir)
# processor.save_pretrained(output_dir)
#
# print(f"Fine-tuning complete. Model saved to {output_dir}")

import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast

# Define simplified class labels
class_mapping = {
    "Zoomorfo (artiodactyla)": "Animal",
    "Zoomorfo (ave)": "Animal",
    "Zoomorfo (piche)": "Animal",
    "Zoomorfo (matuasto)": "Animal",
    "Antropomorfo": "Human",
    "Positivo de mano": "Hand",
    "Negativo de mano": "Hand",
    "Negativo de pata de choique": "Animal_print",
    "Negativo de puño": "Hand",
    "Círculos": "Geometric",
    "Círculos concéntricos": "Geometric",
    "Líneas rectas": "Geometric",
    "Líneas zigzag": "Geometric",
    "Escala": "Other",
    "Persona": "Human",
    "Lazo bola": "Other",
    "Conjuntos de puntos": "Geometric",
    "Impactos": "Other",
    "Tridígitos": "Animal_print"
}

# Create a mapping of simplified labels to numerical IDs
id2label = {i: label for i, label in enumerate(set(class_mapping.values()))}
label2id = {label: i for i, label in id2label.items()}


class CustomDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, processor):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.processor = processor
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not self.image_files:
            raise ValueError(f"No image files found in {image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        ann_path = os.path.join(self.annotation_dir, img_name.rsplit('.', 1)[0] + '.txt')

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"Annotation file not found: {ann_path}")

        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        # Read YOLO annotations and convert to COCO format
        with open(ann_path, 'r') as f:
            annotations = f.readlines()

        coco_annotations = []

        for ann in annotations:
            class_id, x_center, y_center, width, height = map(float, ann.strip().split())
            x_min = (x_center - width / 2) * w
            y_min = (y_center - height / 2) * h
            x_max = (x_center + width / 2) * w
            y_max = (y_center + height / 2) * h

            original_class = list(class_mapping.keys())[int(class_id)]
            simplified_class = class_mapping[original_class]
            category_id = label2id[simplified_class]

            coco_annotations.append({
                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                "category_id": category_id,
                "area": (x_max - x_min) * (y_max - y_min),
                "iscrowd": 0
            })

        if not coco_annotations:
            raise ValueError(f"No valid annotations found for image: {img_name}")

        target = {
            "image_id": idx,
            "annotations": coco_annotations
        }

        # Use the processor to prepare the image and annotations
        encoding = self.processor(images=image, annotations=target, return_tensors="pt")

        # Remove batch dimension for each item in encoding
        for k, v in encoding.items():
            if isinstance(v, torch.Tensor):
                encoding[k] = v.squeeze(0)
            elif isinstance(v, list):
                encoding[k] = v[0] if v else v

        return encoding


def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    encoding = {
        "pixel_values": pixel_values,
        "labels": [{k: v.to(pixel_values.device) for k, v in item["labels"].items()} for item in batch]
    }
    return encoding


# Set up the model and processor
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101",
                                               num_labels=len(id2label),
                                               ignore_mismatched_sizes=True, revision="no_timm")
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101", revision="no_timm")

# Update the model's config with new labels
model.config.id2label = id2label
model.config.label2id = label2id

# Set directories
image_dir = '../data/processed'
annotations_dir = '../data/annotations_processed'
output_dir = '../output/fine_tuned_model'

# Set hyperparameters
batch_size = 16
learning_rate = 1e-4
num_epochs = 50

# Set up the dataset and split into train and validation
try:
    full_dataset = CustomDataset(image_dir, annotations_dir, processor)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Calculate class weights for balanced sampling
    class_counts = {label: 0 for label in label2id.keys()}
    for i in range(len(full_dataset)):
        annotations = full_dataset[i]['labels']['class_labels']
        for ann in annotations:
            class_counts[id2label[ann.item()]] += 1

    total_samples = sum(class_counts.values())
    class_weights = {label: total_samples / count for label, count in class_counts.items()}
    sample_weights = [class_weights[id2label[full_dataset[i]['labels']['class_labels'][0].item()]] for i in
                      range(len(full_dataset))]

    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn,
                                  num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    print(f"Dataset initialized with {len(train_dataset)} training images and {len(val_dataset)} validation images")
except Exception as e:
    print(f"Error initializing dataset: {str(e)}")
    exit(1)

# Set up the optimizer and learning rate scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

# Initialize GradScaler for mixed precision training
scaler = GradScaler()

# Training loop
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        pixel_values = batch["pixel_values"].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

        with autocast():
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

        total_loss += loss.item()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    avg_train_loss = total_loss / len(train_dataloader)

    # Validation loop
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

            with autocast():
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss

            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_dataloader)

    # Update learning rate
    scheduler.step(avg_val_loss)

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

# Save the fine-tuned model
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)

print(f"Fine-tuning complete. Model saved to {output_dir}")

