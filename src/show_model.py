import torch
import argparse
import yaml
import os
from torchinfo import summary  # Import torchinfo for model summary
from utils.utils import load_model  # Use your existing load_model function


# Function to load the YAML configuration
def load_yaml_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


# Function to load the model and weights
def load_model_with_weights(config, model_path, device='cpu'):
    # Load the model architecture from the config
    model = load_model(config['model'])

    # Load the state_dict (weights)
    state_dict = torch.load(model_path, map_location=torch.device(device))

    # Load weights into the model
    model.load_state_dict(state_dict, strict=False)

    # Move the model to the specified device
    model.to(device)

    return model


# Function to simulate inputs for different models
def get_dummy_input(model_type, device):
    if model_type == 'deformabledetr':
        # Deformable DETR expects pixel values and masks
        pixel_values = torch.randn(1, 3, 512, 512).to(device)  # Batch size 1, C=3, H=224, W=224
        pixel_mask = torch.ones(1, 512, 512).to(device)  # Mask the full image
        return {'pixel_values': pixel_values, 'pixel_mask': pixel_mask}

    elif model_type in ['fasterrcnn', 'retinanet']:
        # Faster R-CNN and RetinaNet expect a list of images [C, H, W]
        dummy_image = torch.randn(3, 512, 512).to(device)  # C=3, H=224, W=224
        return [dummy_image]

    else:
        # For other models (if any), return a standard input
        return torch.randn(1, 3, 512, 512).to(device)  # Batch size 1, C=3, H=224, W=224


# Main function to parse arguments and show the model structure
def main():
    parser = argparse.ArgumentParser(description="Load and Display Model Structure")
    parser.add_argument('--config', required=True, help='Path to the config file')
    args = parser.parse_args()

    # Load the YAML config
    config = load_yaml_config(args.config)

    # Retrieve the checkpoint directory from the config
    checkpoint_dir = config.get('checkpoint_dir', None)

    # Model path assumed to be 'last.pth' in checkpoint_dir
    model_path = os.path.join(checkpoint_dir, 'last.pth')

    # Specify device (use CPU for model inspection)
    device = 'cpu'

    # Load the model with weights
    model = load_model_with_weights(config, model_path, device)

    # Get the model type from the config or file name (e.g., deformabledetr, fasterrcnn, etc.)
    model_type = config['model']

    # Get dummy input according to model type
    dummy_input = get_dummy_input(model_type, device)

    # Show the model structure using torchinfo summary
    print("\nDetailed Model Summary (torchinfo):\n")
    summary(model, input_data=dummy_input)


if __name__ == "__main__":
    main()

