# Link Prediction
# LOTO 7/39 (predikcija sledeće grane) - optimizovani sampling

# klasičan ML pristup
# RandomForestClassifier


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from itertools import combinations
import random

# ----------------- LOAD CSV FILE -----------------
file_path = "/Users/milan/Desktop/GHQ/data/loto7_4530_k99.csv"
df = pd.read_csv(file_path, header=None)
df.columns = ['n1','n2','n3','n4','n5','n6','n7']

# ----------------- PREPARE TRAINING DATA -----------------
positive_granas = df.values.tolist()
all_nodes = list(range(1, 40))
negatives = []
neg_sample_size = len(positive_granas)

while len(negatives) < neg_sample_size:
    candidate = sorted(random.sample(all_nodes, 7))
    if candidate not in positive_granas:
        negatives.append(candidate)

X = positive_granas + negatives
y = [1]*len(positive_granas) + [0]*len(negatives)
X_df = pd.DataFrame(X, columns=['n1','n2','n3','n4','n5','n6','n7'])

# ----------------- FEATURE ENGINEERING -----------------
X_features = pd.DataFrame({
    'sum': X_df.sum(axis=1),
    'mean': X_df.mean(axis=1),
    'min': X_df.min(axis=1),
    'max': X_df.max(axis=1),
    'range': X_df.max(axis=1) - X_df.min(axis=1)
})

# ----------------- TRAIN RANDOM FOREST -----------------
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_features, y)

# ----------------- PREDICT NEXT GRANA -----------------
# Sample optimizovan: 500000 nasumičnih kombinacija umesto 1.5M
sample_combs = random.sample(list(combinations(all_nodes, 7)), 500000)
X_pred_df = pd.DataFrame(sample_combs, columns=['n1','n2','n3','n4','n5','n6','n7'])
X_pred_features = pd.DataFrame({
    'sum': X_pred_df.sum(axis=1),
    'mean': X_pred_df.mean(axis=1),
    'min': X_pred_df.min(axis=1),
    'max': X_pred_df.max(axis=1),
    'range': X_pred_df.max(axis=1) - X_pred_df.min(axis=1)
})

probs = clf.predict_proba(X_pred_features)[:,1]
idx_max = np.argmax(probs)
predicted_grana = sorted(sample_combs[idx_max])

print("Predicted next grana (7 nodes):", predicted_grana)
"""
Predicted next grana (7 nodes): [4, 9, 24, 26, 34, 35, 38]

Predicted next grana (7 nodes): [1, 3, 9, 13, 20, 28, 33]
"""