"""
Finds relationship between the frequency of student reasoning and teacher's value-added scores for ELA and Math.

Only plots teachers with at least 5 recorded utterances.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load datasets
reasoning_df = pd.read_csv('student_reasoning - student_reasoning.csv')
va_df = pd.read_csv('ICPSR_36095/DS0004/36095-0004-Data.tsv', sep='\t')

# Calculate reasoning frequency per teacher
reasoning_freq = reasoning_df.groupby('NCTETID')['student_reasoning'].agg([
    ('reasoning_freq', 'mean'),  # Proportion of utterances WITH student reasoning
    ('total_utterances', 'count')  # Total number of utterances per teacher
]).reset_index()

# NOTE: Only considering teachers with at least 5 utterances
reasoning_freq = reasoning_freq[reasoning_freq['total_utterances'] >= 5]

# Join reasoning data with value-added scores data
merged_df = reasoning_freq.merge(
    va_df[['NCTETID', 'STATEVA_E', 'STATEVA_M']],
    on='NCTETID',
    how='inner'
)

# Convert value-added columns to numeric (handles empty strings and invalid values)
merged_df['STATEVA_E'] = pd.to_numeric(merged_df['STATEVA_E'], errors='coerce')
merged_df['STATEVA_M'] = pd.to_numeric(merged_df['STATEVA_M'], errors='coerce')
merged_df = merged_df.dropna(subset=['STATEVA_E', 'STATEVA_M', 'reasoning_freq']) # Remove rows with missing data

# Create figure with two subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ELA plot
ax1 = axes[0]
ax1.scatter(merged_df['reasoning_freq'], merged_df['STATEVA_E'], alpha=0.5, s=10, color='blue')
z1 = np.polyfit(merged_df['reasoning_freq'], merged_df['STATEVA_E'], 1)
ax1.plot(merged_df['reasoning_freq'], np.poly1d(z1)(merged_df['reasoning_freq']), 
         "k-", alpha=0.7, linewidth=1.5)
ax1.set_xlabel('Frequency of Student Reasoning', fontsize=11)
ax1.set_ylabel('ELA Value-Added Score', fontsize=11)
ax1.set_title('ELA Value-Added vs Student Reasoning Frequency', fontsize=12)
ax1.grid(True, alpha=0.2, linestyle='--')

# Math plot
ax2 = axes[1]
ax2.scatter(merged_df['reasoning_freq'], merged_df['STATEVA_M'], alpha=0.5, s=10, color='red')
z2 = np.polyfit(merged_df['reasoning_freq'], merged_df['STATEVA_M'], 1)
ax2.plot(merged_df['reasoning_freq'], np.poly1d(z2)(merged_df['reasoning_freq']), 
         "k-", alpha=0.7, linewidth=1.5)
ax2.set_xlabel('Frequency of Student Reasoning', fontsize=11)
ax2.set_ylabel('Math Value-Added Score', fontsize=11)
ax2.set_title('Math Value-Added vs Student Reasoning Frequency', fontsize=12)
ax2.grid(True, alpha=0.2, linestyle='--')

# Save the figure
plt.tight_layout()
plt.savefig('analyses/reasoning_va_correlation.png', dpi=300, bbox_inches='tight')
print("Plot saved to analyses/reasoning_va_correlation.png")

