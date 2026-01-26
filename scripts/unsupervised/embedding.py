import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import hdbscan
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pyarrow


ncte_single_utterances = pd.read_csv('ncte_single_utterances.csv')
paired_annotations = pd.read_csv('paired_annotations.csv')
student_reasoning = pd.read_csv('student_reasoning.csv')
transcript_metadata = pd.read_csv('transcript_metadata.csv')

df = ncte_single_utterances.merge(
    student_reasoning[['comb_idx', 'NCTETID', 'student_reasoning']],
    on='comb_idx',
    how='left'
)
df = df.merge(
    paired_annotations[['exchange_idx', 'student_on_task', 'teacher_on_task', 'high_uptake', 'focusing_question']].rename(columns={
        'exchange_idx': 'comb_idx',
    }),
    on='comb_idx',
    how='left'
)

student_df = df.loc[
    df['speaker'].isin(['student', 'multiple students'])
].copy()

model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
texts = student_df['text'].tolist()
print(len(texts), "utterances to encode.")

EMB_PATH = "student_embeddings.npy"

if os.path.exists(EMB_PATH):
    embeddings = np.load(EMB_PATH)
else:
    embeddings = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    np.save(EMB_PATH, embeddings)

print("Embeddings shape:", embeddings.shape)

# pca = PCA(n_components=100, random_state=42)
# embeddings_100d = pca.fit_transform(embeddings)

# explained_variance = pca.explained_variance_ratio_.sum()
# print(f"PCA reduced to 100 dimensions, explained variance ratio: {explained_variance:.4f}")


embeddings_100d = embeddings

# Explained variance ratio

# Reduce dimensionality before clustering to speed up HDBSCAN
umap_model = umap.UMAP(
    n_neighbors=50,
    min_dist=0.1,
    n_components=10,
    metric='cosine',
    low_memory=True,
    random_state=None,
    n_jobs=-1  
)

print("Computing UMAP projection for clustering...")
umap_coords = umap_model.fit_transform(embeddings_100d)
print("UMAP-reduced shape:", umap_coords.shape)

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=100,
    min_samples=10,
    metric='euclidean',
    core_dist_n_jobs=os.cpu_count()
)

print("Clustering...")

student_df['cluster'] = clusterer.fit_predict(umap_coords)

print("Number of clusters found:", len(set(student_df['cluster'])) - (1 if -1 in student_df['cluster'].values else 0))



# print dbcv score on samples
sample_indices = np.random.choice(len(umap_coords), size=min(5000, len(umap_coords)), replace=False)
sample_umap_coords = umap_coords[sample_indices]
sample_labels = student_df['cluster'].values[sample_indices]

# Drop noise
mask = sample_labels != -1
sample_umap_coords = sample_umap_coords[mask]
sample_labels = sample_labels[mask]

# Drop clusters with < 2 points
valid_clusters = []
for c in np.unique(sample_labels):
    if np.sum(sample_labels == c) >= 2:
        valid_clusters.append(c)

mask = np.isin(sample_labels, valid_clusters)
sample_umap_coords = sample_umap_coords[mask]
sample_labels = sample_labels[mask]

dbcv_score = hdbscan.validity.validity_index(
    sample_umap_coords.astype('float64'),
    sample_labels
)
print(f"Sample HDBSCAN DBCV score: {dbcv_score:.4f}")

print("Computing 2D UMAP projection for visualization...")
umap_model_plot = umap.UMAP(
    n_neighbors=50,
    min_dist=0.1,
    metric='cosine',
    low_memory=True,
    random_state=None,
    n_jobs=-1
)
umap_coords_2d = umap_model_plot.fit_transform(embeddings_100d)
print("UMAP projection shape:", umap_coords_2d.shape)

student_df['umap_x'] = umap_coords_2d[:, 0]
student_df['umap_y'] = umap_coords_2d[:, 1]

# save student_df
student_df.to_csv('student_df_embeddings.csv', index=False)

clusters = student_df['cluster'].copy()
clusters[clusters == -1] = np.nan

plt.figure(figsize=(10, 7), dpi=200)
scatter = plt.scatter(
    student_df['umap_x'],
    student_df['umap_y'],
    c=clusters,
    s=6,
    cmap='tab20'
)
plt.title("UMAP projection of Student Utterance Embeddings with HDBSCAN Clusters")
plt.savefig("student_utterance_clusters.png", bbox_inches='tight')

reasoning_df = student_df.loc[
    student_df['student_reasoning'].notna()
]

plt.figure(figsize=(10, 7), dpi=200)

sns.scatterplot(
    data=reasoning_df,
    x='umap_x',
    y='umap_y',
    hue='student_reasoning',
    palette='tab10',
    s=10,
    linewidth=0
)

plt.title("UMAP projection of Student Utterances (with Student Reasoning only)")
plt.savefig("student_utterance_reasoning.png", bbox_inches='tight')






student_on_task = student_df.loc[
    student_df['student_on_task'].notna()
]

plt.figure(figsize=(10, 7), dpi=200)

sns.scatterplot(
    data=student_on_task,
    x='umap_x',
    y='umap_y',
    hue='student_on_task',
    palette='tab10',
    s=10,
    linewidth=0
)

plt.title("UMAP projection of Student Utterances (with Student On Task only)")
plt.savefig("student_utterance_student_on_task.png", bbox_inches='tight')






high_uptake = student_df.loc[
    student_df['high_uptake'].notna()
]

plt.figure(figsize=(10, 7), dpi=200)

sns.scatterplot(
    data=high_uptake,
    x='umap_x',
    y='umap_y',
    hue='high_uptake',
    palette='tab10',
    s=10,
    linewidth=0
)

plt.title("UMAP projection of Student Utterances (with High Uptake only)")
plt.savefig("student_utterance_high_uptake.png", bbox_inches='tight')



focusing_question = student_df.loc[
    student_df['focusing_question'].notna()
]

plt.figure(figsize=(10, 7), dpi=200)

sns.scatterplot(
    data=focusing_question,
    x='umap_x',
    y='umap_y',
    hue='focusing_question',
    palette='tab10',
    s=10,
    linewidth=0
)

plt.title("UMAP projection of Student Utterances (with Focusing Question only)")
plt.savefig("student_utterance_focusing_question.png", bbox_inches='tight')