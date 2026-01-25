"""
Finds relationship between the frequency of student reasoning and teacher's experience.

Only plots teachers with at least 5 recorded utterances.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load datasets
reasoning_df = pd.read_csv('student_reasoning - student_reasoning.csv')
teacher_background_df = pd.read_csv('ICPSR_36095/DS0006/36095-0006-Data.tsv', sep='\t')

# Calculate reasoning frequency per teacher
reasoning_freq = reasoning_df.groupby('NCTETID')['student_reasoning'].agg([
    ('reasoning_freq', 'mean'),  # Proportion of utterances WITH student reasoning
    ('total_utterances', 'count')  # Total number of utterances per teacher
]).reset_index()

# NOTE: Only considering teachers with at least 5 utterances
reasoning_freq = reasoning_freq[reasoning_freq['total_utterances'] >= 5]

# Join reasoning data with teacher background data
merged_df = reasoning_freq.merge(
    teacher_background_df[['NCTETID', 'EXPERIENCE']],
    on='NCTETID',
    how='inner'
)

# Convert experience column to numeric (handles empty strings and invalid values)
merged_df['EXPERIENCE'] = pd.to_numeric(merged_df['EXPERIENCE'], errors='coerce')
merged_df = merged_df.dropna(subset=['EXPERIENCE', 'reasoning_freq'])  # Remove rows with missing data

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# Scatter plot
ax.scatter(merged_df['EXPERIENCE'], merged_df['reasoning_freq'], alpha=0.5, s=10, color='blue')
z = np.polyfit(merged_df['EXPERIENCE'], merged_df['reasoning_freq'], 1)
ax.plot(merged_df['EXPERIENCE'], np.poly1d(z)(merged_df['EXPERIENCE']), 
         "k-", alpha=0.7, linewidth=1.5)
ax.set_xlabel('Years of Teaching Experience', fontsize=11)
ax.set_ylabel('Frequency of Student Reasoning', fontsize=11)
ax.set_title('Student Reasoning Frequency vs Teacher Experience', fontsize=12)
ax.grid(True, alpha=0.2, linestyle='--')

# Save the figure
plt.tight_layout()
plt.savefig('analyses/reasoning_experience_correlation.png', dpi=300, bbox_inches='tight')
print("Plot saved to analyses/reasoning_experience_correlation.png")