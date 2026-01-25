#!/usr/bin/env python3
"""
Computes log-odds-ratio using bayes.py and creates a scatter plot visualization.

Usage: python zero-log-odds.py <data_file> <output_chart>
"""

import argparse
import os
import re
import subprocess
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt


def tokenize(text):
    """Lowercase and extract words."""
    return re.findall(r"\b\w+\b", str(text).lower())


def count_words(texts):
    """Count word frequencies across all texts."""
    counts = Counter()
    for text in texts:
        counts.update(tokenize(text))
    return counts


def write_counts(counts, path):
    """Write word counts to file in 'COUNT WORD' format."""
    # Open the file for writing, creating it if it does not exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for word, count in counts.items():
            f.write(f"{count} {word}\n")


def run_bayes(f1_path, f2_path, prior_path):
    """Run bayes.py and parse output into (word, score) tuples."""
    result = subprocess.run(
        ["python3", "scripts/bayes.py", "-f", f1_path, "-s", f2_path, "-p", prior_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"bayes.py failed: {result.stderr}")

    data = []
    for line in result.stdout.strip().split("\n"):
        parts = line.split()
        if len(parts) >= 2:
            try:
                data.append((parts[0], float(parts[-1])))
            except ValueError:
                continue
    return data


def plot_scatter(data, freq_cat1, freq_cat2, output_file, min_freq=5, top_n=50):
    """Create scatter plot with frequency vs log-odds ratio.
    
    Args:
        top_n: Number of top words to display from each category
    """

    # Stopwords to exclude from labels from data/stopwords.txt
    stopwords = set(line.strip() for line in open("stopwords.txt"))

    # Prepare data with frequencies
    all_words = []
    for word, score in data:
        freq = freq_cat1.get(word, 0) if score > 0 else freq_cat2.get(word, 0)
        if freq >= min_freq and word not in stopwords and len(word) > 2:
            is_cat1 = score > 0
            all_words.append((word, score, freq, is_cat1))

    # Split into categories and get top N from each
    cat1_words = [(w, s, f, c) for w, s, f, c in all_words if c]  # Reasoning present (positive scores)
    cat2_words = [(w, s, f, c) for w, s, f, c in all_words if not c]  # No reasoning (negative scores)
    
    # Sort by score: cat1 by highest positive, cat2 by lowest (most negative)
    cat1_words.sort(key=lambda x: -x[1])  # Highest positive first
    cat2_words.sort(key=lambda x: x[1])   # Most negative first
    
    # Take top N from each
    top_cat1 = cat1_words[:top_n]
    top_cat2 = cat2_words[:top_n]
    display_words = set((w, s, f) for w, s, f, _ in top_cat1 + top_cat2)

    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot all points (but only label the top ones)
    cat1_data = [(w, s, f) for w, s, f, c in all_words if c]
    cat2_data = [(w, s, f) for w, s, f, c in all_words if not c]

    if cat1_data:
        ax.scatter(
            [f for _, _, f in cat1_data],
            [s for _, s, _ in cat1_data],
            c="#555555",
            alpha=0.4,
            s=35,
            label="Reasoning present",
            edgecolors="white",
            linewidth=0.3,
        )
    if cat2_data:
        ax.scatter(
            [f for _, _, f in cat2_data],
            [s for _, s, _ in cat2_data],
            c="#6BAED6",
            alpha=0.4,
            s=35,
            label="No reasoning present",
            edgecolors="white",
            linewidth=0.3,
        )

    # Label collision detection - track placed labels as (x, y) in data coordinates
    placed_labels = []
    
    # Get axis ranges for proper scaling
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = xlim[1] - xlim[0] if xlim[1] != xlim[0] else 100
    y_range = ylim[1] - ylim[0] if ylim[1] != ylim[0] else 10
    
    def find_offset(freq, score):
        """Find an offset that avoids overlapping with existing labels.
        Returns (ox, oy, needs_line) where needs_line is True if label is far from point."""
        # Minimum gaps as fraction of axis range - make these large to prevent overlap
        min_x_gap = x_range * 0.08  # 8% of x-axis
        min_y_gap = y_range * 0.06  # 6% of y-axis
        
        # Generate offset options - start with close ones, then go further
        # offsets in points: (x_offset, y_offset)
        close_offsets = [(5, 3), (5, -8), (-45, 3), (-45, -8)]  # Close to point
        far_offsets = []
        for x_off in [8, -70, 80, -120, 130, -170]:  # multiple x positions
            for y_off in range(-80, 90, 15):  # -80 to +80 in steps of 15
                if (x_off, y_off) not in close_offsets:
                    far_offsets.append((x_off, y_off))
        
        # Try close offsets first (no line needed)
        for ox, oy in close_offsets:
            est_x = freq + (ox / 100) * x_range * 0.5
            est_y = score + (oy / 100) * y_range * 0.5
            
            collision = False
            for lx, ly in placed_labels:
                if abs(est_x - lx) < min_x_gap and abs(est_y - ly) < min_y_gap:
                    collision = True
                    break
            
            if not collision:
                placed_labels.append((est_x, est_y))
                return ox, oy, False  # No line needed - close to point
        
        # Try far offsets (line needed)
        for ox, oy in far_offsets:
            est_x = freq + (ox / 100) * x_range * 0.5
            est_y = score + (oy / 100) * y_range * 0.5
            
            collision = False
            for lx, ly in placed_labels:
                if abs(est_x - lx) < min_x_gap and abs(est_y - ly) < min_y_gap:
                    collision = True
                    break
            
            if not collision:
                placed_labels.append((est_x, est_y))
                return ox, oy, True  # Line needed - far from point
        
        # If all positions collide, skip this label
        return None, None, False
    
    # Only label the top words from each category
    for word, score, freq, is_cat1 in top_cat1 + top_cat2:
        ox, oy, needs_line = find_offset(freq, score)
        
        # Skip if no valid position found (would overlap)
        if ox is None:
            continue
            
        color = "#333333" if is_cat1 else "#1a5a8a"
        weight = "bold" if abs(score) > 1.96 else "normal"
        
        # Only draw connecting line if label is far from its point
        arrow_props = dict(arrowstyle="-", color="#aaaaaa", alpha=0.4, linewidth=0.5) if needs_line else None
        
        ax.annotate(
            word, 
            (freq, score), 
            xytext=(ox, oy),
            textcoords='offset points',
            fontsize=7, 
            alpha=0.9, 
            color=color, 
            fontweight=weight,
            arrowprops=arrow_props,
        )

    ax.axhline(0, color="#cccccc", linewidth=1)
    ax.axhline(
        1.96,
        color="green",
        linewidth=0.8,
        linestyle="--",
        alpha=0.4,
        label="Significance (z=±1.96)",
    )
    ax.axhline(-1.96, color="green", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_xlabel("Frequency of Word within Classification", fontsize=12)
    ax.set_ylabel("Weighted Log-Odds-Ratio (Z-score)", fontsize=12)
    ax.set_title(
        "Z-scored Log Odds Ratios (Monroe et al., 2009)\nBold = statistically significant (|z| > 1.96)",
        fontsize=13,
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.savefig(f"data/generated/{output_file}", dpi=150, facecolor="white")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=str, required=True)
    parser.add_argument(
        "--output-chart",
        type=str,
        required=False,
        default=f"data/generated/output_chart.png",
    )
    args = parser.parse_args()

    if not os.path.exists(args.data_file):
        print(f"Error: {args.data_file} not found.")
        exit(1)

    # Load and split data
    df = pd.read_csv(args.data_file)
    reasoning = df[df["student_reasoning"] == 1.0]["text"]
    no_reasoning = df[df["student_reasoning"] == 0.0]["text"]

    # Count words and write temp files
    freq_reasoning = count_words(reasoning)
    freq_no_reasoning = count_words(no_reasoning)
    freq_all = count_words(df["text"])

    write_counts(freq_reasoning, "data/generated/reasoning.txt")
    write_counts(freq_no_reasoning, "data/generated/no_reasoning.txt")
    write_counts(freq_all, "data/generated/prior.txt")

    # Run analysis and plot
    results = run_bayes(
        "data/generated/reasoning.txt",
        "data/generated/no_reasoning.txt",
        "data/generated/prior.txt",
    )
    plot_scatter(results, freq_reasoning, freq_no_reasoning, args.output_chart, 
                 min_freq=2, top_n=30)


if __name__ == "__main__":
    main()
