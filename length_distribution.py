import pickle
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd

# Set paper-quality plotting parameters
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'DejaVu Serif',  # More widely available serif font
    'axes.linewidth': 1.0,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.size': 4,
    'xtick.minor.size': 2,
    'ytick.major.size': 4,
    'ytick.minor.size': 2,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
    'patch.linewidth': 0.5,
    'legend.frameon': True,
    'legend.fontsize': 10,
    'figure.facecolor': 'white'
})

def load_rollout_data(data_dir):
    """Load all rollout data files from the specified directory."""
    rollout_files = glob.glob(os.path.join(data_dir, "rollout_data_*.pt"))
    if not rollout_files:
        print(f"No rollout data files found in {data_dir}")
        return []
    
    rollout_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    print(f"Found {len(rollout_files)} rollout data files in {data_dir}")
    
    all_samples = []
    for file_path in tqdm(rollout_files, desc=f"Loading from {os.path.basename(data_dir)}"):
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Convert dict format back to Sample objects
            samples = []
            for sample_dict in data:
                samples.append(sample_dict)
            
            all_samples.extend(samples)
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    print(f"Total samples loaded from {data_dir}: {len(all_samples)}")
    return all_samples

def analyze_response_lengths(samples, n_samples_per_prompt=4):
    """Analyze response lengths grouped by prompt."""
    
    # Group samples by prompt (every n_samples_per_prompt samples belong to the same prompt)
    prompt_groups = []
    for i in range(0, len(samples), n_samples_per_prompt):
        group = samples[i:i+n_samples_per_prompt]
        if len(group) == n_samples_per_prompt:  # Only include complete groups
            prompt_groups.append(group)
    
    print(f"Found {len(prompt_groups)} complete prompt groups")
    
    # Calculate average response length for each prompt group
    average_lengths = []
    for group in prompt_groups:
        lengths = [sample.get('response_length', 0) for sample in group]
        if all(length > 0 for length in lengths):  # Only consider groups with valid responses
            avg_length = np.mean(lengths)
            average_lengths.append(avg_length)
    
    return average_lengths

def create_paper_figure(dataset_results, output_file="distribution_of_three_dataset.png"):
    """Create a paper-quality figure showing response length distributions."""
    
    # Create figure with appropriate size to match backup image
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Define colors for the three datasets (matching backup image style)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    dataset_names = ['DAPO-Math-17K', 'DeepMath', 'DeepScaler']
    
    # Create histograms with transparency for overlay
    bins = np.linspace(0, 32000, 60)  # More bins to match the backup image
    
    medians = []
    
    for i, (dataset_name, avg_lengths) in enumerate(zip(dataset_names, dataset_results)):
        if avg_lengths:
            ax.hist(avg_lengths, bins=bins, alpha=0.6, color=colors[i], 
                   label=f'{dataset_name} (n={len(avg_lengths):,})', 
                   density=True, edgecolor='white', linewidth=0.5)
            
            # Calculate and store median for this dataset
            median_val = np.median(avg_lengths)
            medians.append((median_val, colors[i], dataset_name))
    
    # Add vertical dashed lines for each dataset's median
    for median_val, color, name in medians:
        ax.axvline(median_val, color=color, linestyle='--', alpha=0.7, linewidth=2)
    
    # Customize the plot to match backup image
    ax.set_xlabel('Average Response Length (tokens)', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.set_title('Distribution of Average Response Lengths Across Datasets', fontsize=16, pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add legend
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Set x-axis to show key values
    ax.set_xlim(0, 30000)
    ax.set_xticks([0, 5000, 10000, 15000, 20000, 25000, 30000])
    ax.set_xticklabels(['0', '5K', '10K', '15K', '20K', '25K', '30K'])
    
    # Improve layout
    plt.tight_layout()
    
    # Save with high DPI for paper quality
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_file.replace('.png', '.pdf'), bbox_inches='tight', 
                facecolor='white')  # Also save as PDF for LaTeX
    
    print(f"Paper figure saved as: {output_file}")
    print(f"PDF version saved as: {output_file.replace('.png', '.pdf')}")
    
    # Print median values for reference
    print("\nDataset Medians:")
    for median_val, _, name in medians:
        print(f"  {name}: {median_val:.0f} tokens")
    
    return fig

def print_dataset_statistics(dataset_results, dataset_names):
    """Print comprehensive statistics for each dataset."""
    
    print("\n" + "="*80)
    print("DATASET COMPARISON STATISTICS")
    print("="*80)
    
    for i, (dataset_name, avg_lengths) in enumerate(zip(dataset_names, dataset_results)):
        if not avg_lengths:
            print(f"\n{dataset_name}: No valid data found")
            continue
            
        print(f"\n{dataset_name}:")
        print(f"  Number of prompt groups: {len(avg_lengths):,}")
        print(f"  Mean average length: {np.mean(avg_lengths):.1f} tokens")
        print(f"  Median average length: {np.median(avg_lengths):.1f} tokens")
        print(f"  Std average length: {np.std(avg_lengths):.1f} tokens")
        print(f"  Min average length: {np.min(avg_lengths):.1f} tokens")
        print(f"  Max average length: {np.max(avg_lengths):.1f} tokens")
        
        # Percentiles
        percentiles = [25, 50, 75, 90, 95, 99]
        print(f"  Percentiles:")
        for p in percentiles:
            val = np.percentile(avg_lengths, p)
            print(f"    {p}th: {val:.1f} tokens")
        
        # Long tail analysis
        long_responses = sum(1 for length in avg_lengths if length > 10000)
        very_long_responses = sum(1 for length in avg_lengths if length > 20000)
        
        print(f"  Long responses (>10K tokens): {long_responses:,} ({long_responses/len(avg_lengths)*100:.1f}%)")
        print(f"  Very long responses (>20K tokens): {very_long_responses:,} ({very_long_responses/len(avg_lengths)*100:.1f}%)")

def main():
    """Main function to create the paper figure."""
    print("Creating paper-quality figure for response length distributions...")
    
    # Define the three data directories
    data_dirs = [
        "/root/dapo-math-17k-inference-data",
        "/root/deepmath_inference_data", 
        "/root/deepscaler-inference-data"
    ]
    
    dataset_names = ['DAPO-Math-17K', 'DeepMath', 'DeepScaler']
    
    # Load and analyze data from all three datasets
    dataset_results = []
    
    for data_dir, dataset_name in zip(data_dirs, dataset_names):
        if os.path.exists(data_dir):
            print(f"\nProcessing {dataset_name}...")
            samples = load_rollout_data(data_dir)
            
            if samples:
                avg_lengths = analyze_response_lengths(samples, n_samples_per_prompt=4)
                dataset_results.append(avg_lengths)
            else:
                print(f"No valid samples found in {data_dir}")
                dataset_results.append([])
        else:
            print(f"Directory not found: {data_dir}")
            dataset_results.append([])
    
    # Check if we have any valid data
    if not any(dataset_results):
        print("No valid data found in any dataset. Please check the data directories.")
        return
    
    # Print statistics
    print_dataset_statistics(dataset_results, dataset_names)
    
    # Create the paper figure
    print("\nCreating paper figure...")
    create_paper_figure(dataset_results)
    
    print("\n" + "="*80)
    print("PAPER FIGURE GENERATION COMPLETE!")
    print("="*80)
    print("Generated file: distribution_of_three_dataset.png")
    print("This figure is ready for inclusion in your ICLR paper.")

if __name__ == "__main__":
    main() 