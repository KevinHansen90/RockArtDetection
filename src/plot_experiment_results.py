import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_experiment_results(csv_path, output_dir):
	"""
	Reads the CSV file containing experiment results and generates PNG plots
	with four subplots for each model, illustrating different metrics over epochs.

	Parameters:
	- csv_path (str): Path to the 'test_results.csv' file.
	- output_dir (str): Directory where the PNG files will be saved.
	"""

	df = pd.read_csv(csv_path)
	print(df)

	# Verify required columns exist
	required_columns = {'Model', 'Technique', 'Epoch', 'Train Loss', 'Val Loss', 'mAP_0.5', 'mar_100'}
	if not required_columns.issubset(df.columns):
		missing = required_columns - set(df.columns)
		raise ValueError(f"The following required columns are missing in the CSV: {missing}")

	# Create output directory if it doesn't exist
	os.makedirs(output_dir, exist_ok=True)

	# Set Seaborn style for better aesthetics
	sns.set(style="whitegrid", context="paper", font_scale=1.2, palette="deep")

	# Get the list of unique models
	models = df['Model'].unique()

	# Define the metrics to plot
	metrics = ['Train Loss', 'Val Loss', 'mAP_0.5', 'mar_100']

	# Iterate over each model to create separate plots
	for model in models:
		model_df = df[df['Model'] == model]
		techniques = model_df['Technique'].unique()

		# Define color palette
		palette = sns.color_palette("tab10", n_colors=len(techniques))
		technique_colors = dict(zip(techniques, palette))

		# Initialize the matplotlib figure
		fig, axs = plt.subplots(2, 2, figsize=(16, 12))
		fig.suptitle(f'Performance Metrics for {model}', fontsize=16, fontweight='bold', y=0.95)

		# Flatten the axs array for easy iteration
		axs = axs.flatten()

		# Iterate over each metric and create a subplot
		for idx, metric in enumerate(metrics):
			ax = axs[idx]
			for technique in techniques:
				tech_df = model_df[model_df['Technique'] == technique].sort_values('Epoch')
				ax.plot(
					tech_df['Epoch'],
					tech_df[metric],
					label=technique,
					color=technique_colors[technique],
					linewidth=2,
					marker='o'  # Adds markers for better visibility
				)
			ax.set_title(metric, fontsize=14, fontweight='semibold')
			ax.set_xlabel('Epoch', fontsize=12)
			ax.set_ylabel(metric, fontsize=12)
			ax.legend(title='Technique', fontsize=10, title_fontsize=12)
			ax.tick_params(axis='both', which='major', labelsize=10)
			ax.grid(True, linestyle='--', alpha=0.7)

		# Adjust layout to prevent overlap
		plt.tight_layout(rect=[0, 0.03, 1, 0.95])

		# Define the output file path
		output_file = os.path.join(output_dir, f"{model}_results.png")

		# Save the figure with high resolution
		plt.savefig(output_file, dpi=300)
		plt.close(fig)  # Close the figure to free memory

		print(f"Saved plot for model '{model}' to '{output_file}'.")


if __name__ == "__main__":
	# Define the path to your CSV file
	csv_file_path = 'test_results.csv'  # Update this path if necessary

	# Define the directory where plots will be saved
	plots_output_dir = '../experiment_plots'  # You can change this as needed

	# Generate the plots
	plot_experiment_results(csv_file_path, plots_output_dir)
