# Link Prediction
# LOTO 7/39 (predikcija sledeće grane) - optimizovani sampling

# klasičan ML pristup
# RandomForestClassifier



# https://graphsinspace.net
# https://tigraphs.pmf.uns.ac.rs



import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import random

# ----------------- LOAD CSV FILE -----------------
SEED = 39
random.seed(SEED)
np.random.seed(SEED)

file_path = "/Users/4c/Desktop/GHQ/data/loto7_4586_k24.csv"
df = pd.read_csv(file_path, header=None)
df.columns = ['n1','n2','n3','n4','n5','n6','n7']

# ----------------- PREPARE TRAINING DATA -----------------
positive_granas = [tuple(sorted(row)) for row in df.values.tolist()]
positive_set = set(positive_granas)
all_nodes = list(range(1, 40))
negatives = []
neg_sample_size = len(positive_granas)

# v2: membership je preko set-a (mnogo brže od list in)
while len(negatives) < neg_sample_size:
    candidate = tuple(sorted(random.sample(all_nodes, 7)))
    if candidate not in positive_set:
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
print("[start] Trening RandomForestClassifier...")
clf = RandomForestClassifier(n_estimators=220, random_state=SEED, n_jobs=-1)
clf.fit(X_features, y)
print("[ok] Model istreniran. Kreće sampling kandidata...\n")

# ----------------- PREDICT NEXT GRANA -----------------
# Sample optimizovan: 500000 nasumičnih kombinacija bez pravljenja list(combinations(...)) u RAM
target_samples = 500000
batch_size = 50000
progress_every = 2  # po batch-u

sample_combs = set()
while len(sample_combs) < target_samples:
    sample_combs.add(tuple(sorted(random.sample(all_nodes, 7))))
sample_combs = list(sample_combs)

best_prob = -1.0
predicted_grana = None

for bi, start in enumerate(range(0, len(sample_combs), batch_size), start=1):
    batch = sample_combs[start:start+batch_size]
    X_pred_df = pd.DataFrame(batch, columns=['n1','n2','n3','n4','n5','n6','n7'])
    X_pred_features = pd.DataFrame({
        'sum': X_pred_df.sum(axis=1),
        'mean': X_pred_df.mean(axis=1),
        'min': X_pred_df.min(axis=1),
        'max': X_pred_df.max(axis=1),
        'range': X_pred_df.max(axis=1) - X_pred_df.min(axis=1)
    })

    probs = clf.predict_proba(X_pred_features)[:, 1]
    idx_max = int(np.argmax(probs))
    if probs[idx_max] > best_prob:
        best_prob = float(probs[idx_max])
        predicted_grana = sorted(batch[idx_max])

    if bi == 1 or bi % progress_every == 0:
        print(f"[progress] batch={bi} best_prob={best_prob:.6f} best_grana={predicted_grana}")

print("Predicted next grana (7 nodes):", predicted_grana)

"""
[start] Trening RandomForestClassifier...
[ok] Model istreniran. Kreće sampling kandidata...

[progress] batch=1 best_prob=1.000000 best_grana=[5, 12, 15, 22, x, y, z]
[progress] batch=2 best_prob=1.000000 best_grana=[5, 12, 15, 22, x, y, z]
[progress] batch=4 best_prob=1.000000 best_grana=[5, 12, 15, 22, x, y, z]
[progress] batch=6 best_prob=1.000000 best_grana=[5, 12, 15, 22, x, y, z]
[progress] batch=8 best_prob=1.000000 best_grana=[5, 12, 15, 22, x, y, z]
[progress] batch=10 best_prob=1.000000 best_grana=[5, 12, 15, 22, x, y, z]
Predicted next grana (7 nodes): [5, 12, 15, 22, x, y, z]
"""



"""
brži negative sampling (set umesto list membership),
jači model (RandomForestClassifier, n_estimators=220, n_jobs=-1, seed 39),
bez list(combinations(...)) koje ubija RAM,
batch scoring sa progres ispisom (best_prob, best_grana)
"""
