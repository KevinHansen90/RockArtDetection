import subprocess
import yaml

# Define the config files and corresponding model names
train_script = "train.py"
config_files = {
	"configs/config_faster_rcnn.yaml": "fasterrcnn",
	"configs/config_retinanet.yaml": "retinanet",
	"configs/config_deformable_detr.yaml": "deformabledetr"
}

# Define the list of train_transforms values to test
train_transforms_list = ["none", "clahe", "laplacian_pyramid", "bilateral_filter", "unsharp_masking"]


# Function to update train_transforms and checkpoint_dir in the YAML config file
def update_config(config_file, model_name, train_transforms_value):
	# Load the config YAML file
	with open(config_file, 'r') as file:
		config = yaml.safe_load(file)

	# Update the train_transforms parameter
	config['train_transforms'] = train_transforms_value

	# Update the checkpoint_dir parameter
	checkpoint_dir = f"/Users/kevinhansen/Documents/Git/RockArtDetection/src/outputs/{model_name}/{train_transforms_value}"
	config['checkpoint_dir'] = checkpoint_dir

	# Write the updated config back to the YAML file
	with open(config_file, 'w') as file:
		yaml.dump(config, file)


# Loop through each train_transforms value and config file
for train_transforms in train_transforms_list:
	for config_file, model_name in config_files.items():
		# Update the train_transforms and checkpoint_dir parameters in the YAML file
		update_config(config_file, model_name, train_transforms)

		# Run the training command with the updated config
		print(f"Running {train_script} with {config_file} and train_transforms={train_transforms}")
		subprocess.run(['python', train_script, '--config', config_file])

print("All training sessions completed.")
