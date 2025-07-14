#!/usr/bin/env python3
"""
Script to analyze rollout data and visualize response length statistics.
Enhanced with within-group correlation analysis and detailed explanations.
"""

import pickle
import os
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from scipy import stats
from itertools import combinations

def load_rollout_data(data_dir="/root/inference_data"):
    """Load all rollout data files from the specified directory."""
    rollout_files = glob.glob(os.path.join(data_dir, "rollout_data_*.pt"))
    rollout_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    print(f"Found {len(rollout_files)} rollout data files")
    
    all_samples = []
    for file_path in tqdm(rollout_files, desc="Loading rollout data"):
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
    
    print(f"Total samples loaded: {len(all_samples)}")
    return all_samples

def analyze_within_group_correlations(prompt_stats):
    """分析组内response长度的相关性"""
    print("\n=== 组内相关性分析 ===")
    
    # 计算每个组内response长度的相关系数
    within_group_correlations = []
    group_variances = []
    
    for group_idx, stats in prompt_stats.items():
        lengths = stats['lengths']
        if len(lengths) == 4:  # 确保有4个response
            # 计算组内方差
            group_var = np.var(lengths)
            group_variances.append(group_var)
            
            # 计算组内所有对之间的相关性（虽然对于4个数据点相关性不太有意义，但可以看趋势）
            # 这里我们用变异系数来衡量组内一致性
            cv = np.std(lengths) / np.mean(lengths) if np.mean(lengths) > 0 else 0
            within_group_correlations.append(cv)
    
    # 计算整体统计
    mean_within_group_cv = np.mean(within_group_correlations)
    mean_group_variance = np.mean(group_variances)
    
    # 计算所有response长度的总体方差
    all_lengths = []
    for stats in prompt_stats.values():
        all_lengths.extend(stats['lengths'])
    total_variance = np.var(all_lengths)
    
    print(f"组内平均变异系数: {mean_within_group_cv:.4f}")
    print(f"组内平均方差: {mean_group_variance:.2f}")
    print(f"总体方差: {total_variance:.2f}")
    print(f"组内方差占总体方差比例: {mean_group_variance/total_variance:.2%}")
    
    return {
        'within_group_cvs': within_group_correlations,
        'group_variances': group_variances,
        'mean_within_group_cv': mean_within_group_cv,
        'mean_group_variance': mean_group_variance,
        'total_variance': total_variance
    }

def analyze_response_lengths(samples, n_samples_per_prompt=4):
    """Analyze response lengths grouped by prompt."""
    print(f"Analyzing with {n_samples_per_prompt} samples per prompt...")
    
    # Group samples by prompt (every n_samples_per_prompt samples belong to the same prompt)
    prompt_groups = []
    for i in range(0, len(samples), n_samples_per_prompt):
        group = samples[i:i+n_samples_per_prompt]
        if len(group) == n_samples_per_prompt:  # Only include complete groups
            prompt_groups.append(group)
    
    print(f"Found {len(prompt_groups)} complete prompt groups")
    
    # Calculate statistics for each prompt group
    prompt_stats = {}
    for group_idx, group in enumerate(prompt_groups):
        lengths = [sample.get('response_length', 0) for sample in group]
        rewards = [sample.get('reward', 0) for sample in group]
        
        if any(length > 0 for length in lengths):  # Only consider groups with valid responses
            prompt_stats[group_idx] = {
                'mean': np.mean(lengths),
                'median': np.median(lengths),
                'std': np.std(lengths),
                'min': np.min(lengths),
                'max': np.max(lengths),
                'count': len(lengths),
                'lengths': lengths,
                'rewards': rewards,
                'mean_reward': np.mean(rewards),
                'cv': np.std(lengths) / np.mean(lengths) if np.mean(lengths) > 0 else 0,  # 变异系数
                'prompt_preview': group[0]['prompt'][:200] + "..." if len(group[0]['prompt']) > 200 else group[0]['prompt']
            }
    
    return prompt_stats

def create_visualizations(prompt_stats, correlation_analysis):
    """Create various visualizations for the response length analysis."""
    
    print("\n=== 图表说明 ===")
    
    # Prepare data for plotting
    prompt_ids = list(prompt_stats.keys())
    mean_lengths = [prompt_stats[pid]['mean'] for pid in prompt_ids]
    std_lengths = [prompt_stats[pid]['std'] for pid in prompt_ids]
    mean_rewards = [prompt_stats[pid]['mean_reward'] for pid in prompt_ids]
    cvs = [prompt_stats[pid]['cv'] for pid in prompt_ids]
    
    # Set up the plotting style
    plt.style.use('default')
    fig = plt.figure(figsize=(24, 20))
    
    # 1. Average response length per prompt (line plot)
    plt.subplot(3, 2, 1)
    plt.plot(prompt_ids, mean_lengths, alpha=0.7, color='skyblue', linewidth=1)
    plt.scatter(prompt_ids, mean_lengths, alpha=0.5, s=10, color='darkblue')
    plt.xlabel('Prompt ID')
    plt.ylabel('Average Response Length (tokens)')
    plt.title('1. Average Response Length per Prompt')
    plt.grid(True, alpha=0.3)
    
    print("图1 - 每个Prompt的平均Response长度:")
    print("  显示每个Prompt的4个response的平均长度")
    print("  横轴：Prompt ID（0-17279）")
    print("  纵轴：平均response长度（tokens）")
    print("  用途：观察不同Prompt的难度分布和模型响应长度趋势")
    
    # 2. Distribution of average response lengths
    plt.subplot(3, 2, 2)
    plt.hist(mean_lengths, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.xlabel('Average Response Length (tokens)')
    plt.ylabel('Number of Prompts')
    plt.title('2. Distribution of Average Response Lengths')
    plt.axvline(np.mean(mean_lengths), color='red', linestyle='--', 
                label=f'Overall Mean: {np.mean(mean_lengths):.1f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    print("\n图2 - 平均Response长度分布:")
    print("  显示所有Prompt的平均response长度的分布情况")
    print("  横轴：平均response长度（tokens）")
    print("  纵轴：该长度区间的Prompt数量")
    print("  用途：了解数据集中response长度的整体分布特征")
    # 3. Response length vs reward relationship
    plt.subplot(3, 2, 3)
    colors = ['red' if reward < 0.5 else 'green' for reward in mean_rewards]
    plt.scatter(mean_lengths, mean_rewards, alpha=0.6, s=30, c=colors)
    plt.xlabel('Average Response Length (tokens)')
    plt.ylabel('Average Reward')
    plt.title('3. Response Length vs Reward')
    plt.grid(True, alpha=0.3)
    
    print("\n图3 - Response长度与奖励关系:")
    print("  显示response长度与奖励的关系")
    print("  横轴：平均response长度")
    print("  纵轴：平均奖励值")
    print("  颜色：红色=低奖励(<0.5)，绿色=高奖励(≥0.5)")
    print("  用途：分析长response是否更容易获得高奖励")
    
    # 4. Within-group coefficient of variation
    plt.subplot(3, 2, 4)
    plt.hist(cvs, bins=30, alpha=0.7, color='purple', edgecolor='black')
    plt.xlabel('Coefficient of Variation')
    plt.ylabel('Number of Prompts')
    plt.title('4. Within-Group Length Consistency')
    plt.axvline(np.mean(cvs), color='red', linestyle='--', 
                label=f'Mean CV: {np.mean(cvs):.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    print("\n图4 - 组内长度一致性:")
    print("  显示每个Prompt组内4个response长度的一致性")
    print("  横轴：变异系数（标准差/平均值）")
    print("  纵轴：该变异系数的Prompt数量")
    print("  用途：评估同一Prompt的不同response长度是否相近")

    # 5. CV vs Mean Length
    plt.subplot(3, 2, 5)
    plt.scatter(mean_lengths, cvs, alpha=0.6, s=30, color='brown')
    plt.xlabel('Average Response Length (tokens)')
    plt.ylabel('Coefficient of Variation')
    plt.title('5. Length Consistency vs Average Length')
    plt.grid(True, alpha=0.3)
    
    cv_length_corr = np.corrcoef(mean_lengths, cvs)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {cv_length_corr:.3f}', 
             transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    print("\n图5 - 长度一致性vs平均长度:")
    print("  显示Prompt平均长度与组内一致性的关系")
    print("  横轴：平均response长度")
    print("  纵轴：变异系数（越小越一致）")
    print("  用途：分析长Prompt是否更容易产生长度不一致的response")
    
    # 6. Within-group variance analysis
    plt.subplot(3, 2, 6)
    group_variances = correlation_analysis['group_variances']
    plt.hist(group_variances, bins=30, alpha=0.7, color='teal', edgecolor='black')
    plt.xlabel('Within-Group Variance')
    plt.ylabel('Number of Prompts')
    plt.title('6. Within-Group Variance Distribution')
    plt.axvline(correlation_analysis['mean_group_variance'], color='red', linestyle='--', 
                label=f'Mean: {correlation_analysis["mean_group_variance"]:.1f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    print("\n图6 - 组内方差分布:")
    print("  显示每个Prompt组内4个response长度方差的分布")
    print("  横轴：组内方差")
    print("  纵轴：该方差值的Prompt数量")
    print("  用途：量化同一Prompt内response长度的分散程度")
    
    plt.tight_layout()
    plt.savefig('rollout_response_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n可视化图表已保存为 'rollout_response_analysis.png'")
    
    return fig

def print_summary_statistics(prompt_stats, correlation_analysis):
    """Print summary statistics of the analysis."""
    all_lengths = []
    all_rewards = []
    for stats in prompt_stats.values():
        all_lengths.extend(stats['lengths'])
        all_rewards.extend(stats['rewards'])
    
    mean_lengths = [stats['mean'] for stats in prompt_stats.values()]
    
    print("\n" + "="*80)
    print("ROLLOUT DATA ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"Total number of prompts: {len(prompt_stats):,}")
    print(f"Total number of responses: {len(all_lengths):,}")
    print(f"Samples per prompt: {len(all_lengths) // len(prompt_stats)}")
    
    print("\nOverall Response Length Statistics:")
    print(f"  Mean: {np.mean(all_lengths):.2f} tokens")
    print(f"  Median: {np.median(all_lengths):.2f} tokens")
    print(f"  Std: {np.std(all_lengths):.2f} tokens")
    print(f"  Min: {np.min(all_lengths)} tokens")
    print(f"  Max: {np.max(all_lengths)} tokens")
    
    print("\nPrompt-wise Average Response Length Statistics:")
    print(f"  Mean of averages: {np.mean(mean_lengths):.2f} tokens")
    print(f"  Median of averages: {np.median(mean_lengths):.2f} tokens")
    print(f"  Std of averages: {np.std(mean_lengths):.2f} tokens")
    print(f"  Min average: {np.min(mean_lengths):.2f} tokens")
    print(f"  Max average: {np.max(mean_lengths):.2f} tokens")
    
    print("\nWithin-Group Consistency Analysis:")
    print(f"  组内平均变异系数: {correlation_analysis['mean_within_group_cv']:.4f}")
    print(f"  组内平均方差: {correlation_analysis['mean_group_variance']:.2f}")
    print(f"  总体方差: {correlation_analysis['total_variance']:.2f}")
    print(f"  组内方差占总体方差比例: {correlation_analysis['mean_group_variance']/correlation_analysis['total_variance']:.2%}")
    
    # 解释变异系数
    if correlation_analysis['mean_within_group_cv'] < 0.2:
        consistency_desc = "高度一致"
    elif correlation_analysis['mean_within_group_cv'] < 0.5:
        consistency_desc = "中等一致"
    else:
        consistency_desc = "低一致性"
    
    print(f"  组内一致性评价: {consistency_desc}")
    
    print("\nReward Statistics:")
    unique_rewards = set(all_rewards)
    for reward in sorted(unique_rewards):
        count = all_rewards.count(reward)
        percentage = (count / len(all_rewards)) * 100
        print(f"  Reward {reward}: {count:,} samples ({percentage:.1f}%)")
    
    # Find prompts with most/least variability
    variabilities = [(pid, stats['cv']) for pid, stats in prompt_stats.items()]
    variabilities.sort(key=lambda x: x[1])
    
    print("\nPrompts with highest consistency (lowest CV):")
    for i in range(min(5, len(variabilities))):
        pid, cv = variabilities[i]
        mean_len = prompt_stats[pid]['mean']
        mean_reward = prompt_stats[pid]['mean_reward']
        print(f"  Prompt {pid}: mean={mean_len:.1f}, CV={cv:.3f}, reward={mean_reward:.2f}")
    
    print("\nPrompts with lowest consistency (highest CV):")
    for i in range(min(5, len(variabilities))):
        pid, cv = variabilities[-(i+1)]
        mean_len = prompt_stats[pid]['mean']
        mean_reward = prompt_stats[pid]['mean_reward']
        print(f"  Prompt {pid}: mean={mean_len:.1f}, CV={cv:.3f}, reward={mean_reward:.2f}")

def save_detailed_results(prompt_stats, correlation_analysis, output_file="rollout_analysis_detailed.csv"):
    """Save detailed analysis results to CSV file."""
    
    data = []
    for prompt_id, stats in prompt_stats.items():
        data.append({
            'prompt_id': prompt_id,
            'mean_length': stats['mean'],
            'median_length': stats['median'],
            'std_length': stats['std'],
            'cv': stats['cv'],
            'min_length': stats['min'],
            'max_length': stats['max'],
            'sample_count': stats['count'],
            'mean_reward': stats['mean_reward'],
            'individual_lengths': ','.join(map(str, stats['lengths'])),
            'individual_rewards': ','.join(map(str, stats['rewards'])),
            'prompt_preview': stats['prompt_preview']
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('prompt_id')
    df.to_csv(output_file, index=False)
    
    print(f"\nDetailed results saved to: {output_file}")
    print("CSV文件包含每个Prompt的详细统计信息和4个response的具体长度")

def main():
    """Main function to run the analysis."""
    print("Starting enhanced rollout data analysis...")
    print("本分析将检查组内response长度的相关性并详细解释所有图表")
    
    # Load all rollout data
    samples = load_rollout_data()
    
    if not samples:
        print("No samples found. Please check the data directory.")
        return
    
    # Analyze response lengths (assuming 4 samples per prompt)
    prompt_stats = analyze_response_lengths(samples, n_samples_per_prompt=4)
    
    if not prompt_stats:
        print("No valid prompt statistics found.")
        return
    
    # Analyze within-group correlations
    correlation_analysis = analyze_within_group_correlations(prompt_stats)
    
    # Print summary statistics
    print_summary_statistics(prompt_stats, correlation_analysis)
    
    # Create visualizations
    print("\nCreating enhanced visualizations...")
    create_visualizations(prompt_stats, correlation_analysis)
    
    # Save detailed results
    save_detailed_results(prompt_stats, correlation_analysis)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("Generated files:")
    print("  - rollout_response_analysis.png (enhanced visualization with 9 charts)")
    print("  - rollout_analysis_detailed.csv (detailed statistics)")
    print("\n关键发现将在上述统计信息中显示")

if __name__ == "__main__":
    main() 