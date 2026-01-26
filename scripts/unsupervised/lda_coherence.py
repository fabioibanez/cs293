import matplotlib.pyplot as plt
import pandas as pd

# Number of topics
k_vals = [2, 3, 4, 5, 6, 7, 8, 9, 10]

# Coherence values from your MALLET runs
coherence_scores = pd.read_csv('mallet_lda_coherence_scores.csv')
cv_scores = coherence_scores['coherence_cv'].tolist()
plt.figure(figsize=(8, 5))

plt.plot(k_vals, cv_scores, marker='o', label='c_v coherence')
# plt.plot(k_vals, umass_scores, marker='o', label='UMass coherence')
# plt.plot(k_vals, uci_scores, marker='o', label='UCI coherence')
# plt.plot(k_vals, npmi_scores, marker='o', label='NPMI coherence')
plt.xlabel('Number of Topics (k)')
plt.ylabel('Coherence Score')
plt.title('MALLET LDA Coherence vs Number of Topics')
#plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('mallet_lda_coherence.png')
plt.show()
