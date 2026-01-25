"""
Finds relationship between the frequency of student reasoning and classroom observation scores.

Groups observation data by teacher, averages score columns, and correlates with student reasoning frequency.
Only plots teachers with at least 5 recorded utterances.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

# Load datasets
reasoning_df = pd.read_csv('student_reasoning - student_reasoning.csv')
obs_df = pd.read_csv('ICPSR_36095/DS0001/36095-0001-Data.tsv', sep='\t')

# Process observation data: group by NCTETID and average score columns
clpc_index = obs_df.columns.get_loc('CLPC')
score_columns = obs_df.columns[clpc_index:].tolist()
obs_df_filtered = obs_df[['NCTETID'] + score_columns]

# Replace 998 (missing data code) with NaN before averaging
for col in score_columns:
    obs_df_filtered[col] = obs_df_filtered[col].replace(998, np.nan)

# Group by NCTETID and average score columns
obs_avg = obs_df_filtered.groupby('NCTETID')[score_columns].mean().reset_index()

# Calculate reasoning frequency per teacher
reasoning_freq = reasoning_df.groupby('NCTETID')['student_reasoning'].agg([
    ('reasoning_freq', 'mean'),  # Proportion of utterances WITH student reasoning
    ('total_utterances', 'count')  # Total number of utterances per teacher
]).reset_index()

# NOTE: Only considering teachers with at least 5 utterances
reasoning_freq = reasoning_freq[reasoning_freq['total_utterances'] >= 5]

# Join reasoning data with observation scores
merged_df = reasoning_freq.merge(obs_avg, on='NCTETID', how='inner')

# Remove rows with missing data
merged_df = merged_df.dropna(subset=['reasoning_freq'] + score_columns)

# Mapping from score codes to scoring criteria items
score_names = {
    'CLPC': 'Positive Climate',
    'CLNC': 'Negative Climate',
    'CLTS': 'Teacher Sensitivity',
    'CLRSP': 'Regard for Student Perspectives',
    'CLBM': 'Behavior Management',
    'CLPRDT': 'Productivity',
    'CLILF': 'Instructional Learning Formats',
    'CLCU': 'Content Understanding',
    'CLAPS': 'Analysis and Inquiry',
    'CLQF': 'Quality of Feedback',
    'CLINSTD': 'Instructional Dialogue',
    'CLSTENG': 'Student Engagement'
}

# Create a grid of scatter plots
n_scores = len(score_columns)
n_cols = 4
n_rows = (n_scores + n_cols - 1) // n_cols  # Ceiling division

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
if n_rows == 1:
    axes = axes if isinstance(axes, np.ndarray) else [axes]
else:
    axes = axes.flatten()

for idx, score_col in enumerate(score_columns):
    ax = axes[idx]
    
    # Get valid data for this score
    valid_data = merged_df[[score_col, 'reasoning_freq']].dropna()
    
    if len(valid_data) > 0:
        # Calculate correlation and p-value
        corr, p = pearsonr(valid_data['reasoning_freq'], valid_data[score_col])
        
        # Color coding: red for p < 0.05, darker orange for 0.05 <= p < 0.1, gray otherwise
        if p < 0.05:
            point_color = 'red'
        elif p < 0.1:
            point_color = '#CC6600'  # Darker orange
        else:
            point_color = 'gray'
        
        # Scatter plot
        ax.scatter(valid_data['reasoning_freq'], valid_data[score_col], 
                  alpha=0.5, s=10, color=point_color)
        
        # Regression line
        z = np.polyfit(valid_data['reasoning_freq'], valid_data[score_col], 1)
        ax.plot(valid_data['reasoning_freq'], np.poly1d(z)(valid_data['reasoning_freq']), 
                "k-", alpha=0.7, linewidth=1.5)
        
        # Title with full name and p-value
        score_name = score_names.get(score_col, score_col)
        ax.set_xlabel('Frequency of Student Reasoning', fontsize=9)
        ax.set_ylabel('Avg Score', fontsize=9)
        ax.set_title(f'{score_name}\n(p={p:.3f})', fontsize=10)
        ax.grid(True, alpha=0.2, linestyle='--')
    else:
        score_name = score_names.get(score_col, score_col)
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(score_name, fontsize=10)


plt.tight_layout()
plt.savefig('analyses/reasoning_observation_correlation.png', dpi=300, bbox_inches='tight')
print(f"\nPlot saved to analyses/reasoning_observation_correlation.png")
