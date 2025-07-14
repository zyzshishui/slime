#!/usr/bin/env python3
"""
Script to filter the original dataset based on response length statistics.
Selects top 25% and bottom 25% by average response length.
"""

import pandas as pd
import json
import numpy as np
from tqdm import tqdm

def load_statistics(csv_file="rollout_analysis_detailed.csv"):
    """Load the detailed statistics from CSV file."""
    print(f"Loading statistics from {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"Loaded statistics for {len(df)} prompts")
    return df

def calculate_thresholds(df):
    """Calculate the 25th and 97th percentile thresholds."""
    mean_lengths = df['mean_length'].values
    
    # Calculate percentiles
    p25 = np.percentile(mean_lengths, 25)
    p97 = np.percentile(mean_lengths, 97)
    
    print(f"\nLength distribution statistics:")
    print(f"  Min: {np.min(mean_lengths):.1f} tokens")
    print(f"  25th percentile: {p25:.1f} tokens")
    print(f"  Median: {np.median(mean_lengths):.1f} tokens")
    print(f"  97th percentile: {p97:.1f} tokens")
    print(f"  Max: {np.max(mean_lengths):.1f} tokens")
    
    return p25, p97

def filter_prompts(df, p25, p9):
    """Filter prompts to get top 25% and bottom 25% by mean length."""
    
    # Bottom 25% (shortest responses)
    bottom_25_mask = df['mean_length'] <= p25
    bottom_25_prompts = df[bottom_25_mask]['prompt_id'].values
    
    # Top 25% (longest responses)
    top_25_mask = df['mean_length'] >= p97
    top_25_prompts = df[top_25_mask]['prompt_id'].values
    
    print(f"\nFiltered prompts:")
    print(f"  Bottom 25% (≤{p25:.1f} tokens): {len(bottom_25_prompts)} prompts")
    print(f"  Top 25% (≥{p97:.1f} tokens): {len(top_25_prompts)} prompts")
    print(f"  Total selected: {len(bottom_25_prompts) + len(top_25_prompts)} prompts")
    
    # Combine the prompt IDs
    selected_prompt_ids = set(bottom_25_prompts) | set(top_25_prompts)
    
    return selected_prompt_ids, len(bottom_25_prompts), len(top_25_prompts)

def load_original_data(input_file="/root/dapo-math-17k/dapo-math-17k.jsonl"):
    """Load the original JSONL dataset."""
    print(f"\nLoading original dataset from {input_file}...")
    
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"Loaded {len(data)} samples from original dataset")
    return data

def filter_and_save_data(data, selected_prompt_ids, output_file="/root/dapo-math-17k/dapo-math-17k_filtered_25-97.jsonl"):
    """Filter the original data and save to new file."""
    print(f"\nFiltering and saving data...")
    
    filtered_data = []
    
    # Convert selected_prompt_ids to a sorted list for better performance
    selected_indices = sorted(list(selected_prompt_ids))
    
    print(f"Selected prompt indices range: {min(selected_indices)} to {max(selected_indices)}")
    
    for prompt_id in tqdm(selected_indices, desc="Filtering samples"):
        if prompt_id < len(data):
            filtered_data.append(data[prompt_id])
        else:
            print(f"Warning: prompt_id {prompt_id} is out of range (dataset has {len(data)} samples)")
    
    # Save filtered data
    print(f"Saving {len(filtered_data)} samples to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in filtered_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Successfully saved filtered dataset to {output_file}")
    return len(filtered_data)

def print_summary(original_count, filtered_count, bottom_count, top_count, p25, p9):
    """Print summary of the filtering process."""
    print("\n" + "="*80)
    print("DATA FILTERING SUMMARY")
    print("="*80)
    
    print(f"Original dataset: {original_count:,} samples")
    print(f"Filtered dataset: {filtered_count:,} samples")
    print(f"Filtering ratio: {(filtered_count/original_count)*100:.1f}%")
    
    print(f"\nLength thresholds:")
    print(f"  Bottom 25% threshold: ≤{p25:.1f} tokens ({bottom_count:,} prompts)")
    print(f"  Top 25% threshold: ≥{p97:.1f} tokens ({top_count:,} prompts)")
    
    print(f"\nFiltered data includes:")
    print(f"  - Shortest response prompts (requiring concise answers)")
    print(f"  - Longest response prompts (requiring detailed reasoning)")
    print(f"  - Good coverage of both simple and complex mathematical problems")
    
    print(f"\nData quality:")
    print(f"  - Excludes middle 50% of prompts with moderate response lengths")
    print(f"  - Focuses on extreme cases for better training contrast")
    print(f"  - Maintains diversity in problem types and difficulty levels")

def verify_filtering(input_file, output_file):
    """Verify the filtering results by checking a few samples."""
    print(f"\nVerifying filtered data...")
    
    # Read a few samples from both files for comparison
    with open(input_file, 'r', encoding='utf-8') as f:
        original_first = json.loads(f.readline())
    
    with open(output_file, 'r', encoding='utf-8') as f:
        filtered_first = json.loads(f.readline())
    
    print(f"Original sample structure: {list(original_first.keys())}")
    print(f"Filtered sample structure: {list(filtered_first.keys())}")
    print(f"Structure match: {list(original_first.keys()) == list(filtered_first.keys())}")

def main():
    """Main function to run the data filtering process."""
    print("Starting data filtering based on response length statistics...")
    
    try:
        # Load statistics
        df = load_statistics()
        
        # Calculate thresholds
        p25, p97
 = calculate_thresholds(df)
        
        # Filter prompts
        selected_prompt_ids, bottom_count, top_count = filter_prompts(df, p25, p97
)
        
        # Load original data
        original_data = load_original_data()
        
        # Filter and save data
        filtered_count = filter_and_save_data(original_data, selected_prompt_ids)
        
        # Print summary
        print_summary(len(original_data), filtered_count, bottom_count, top_count, p25, p97
)
        
        # Verify filtering results
        verify_filtering("/root/dapo-math-17k/dapo-math-17k.jsonl", "/root/dapo-math-17k/dapo-math-17k_filtered.jsonl")
        
        print("\nData filtering completed successfully!")
        
    except Exception as e:
        print(f"Error during data filtering: {e}")
        raise

if __name__ == "__main__":
    main() 