# LOTO 7/39 - paralelizovana batch predikcija sledeće grane

# RandomForestRegressor

# paralelizovanu verziju koja koristi ThreadPoolExecutor 
# da obrađuje batch-eve istovremeno po više CPU jezgara 
# i ubrza predikciju sledeće grane.

"""
Još bržu verziju sa paralelizacijom po CPU jezgrima, 
da batch-eve obrađuje istovremeno i da završi u realnijem vremenu

Prednosti ove verzije:
Koristi više CPU jezgra za obradu batch-eva istovremeno.
Batch-ovi su veličine 500k, ali mogu se prilagoditi memoriji.
Vraća jednu najverovatniju sledeću granu direktno iz svih 15,380,937 kombinacija.
Ova verzija bi trebalo da značajno ubrza predikciju 
u odnosu na sekvencijalnu batch obradu.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------- LOAD CSV FILE -----------------
file_path = "/data/loto7_4530_k99.csv"
df = pd.read_csv(file_path, header=None)
df.columns = ['n1','n2','n3','n4','n5','n6','n7']

num_nodes = 39
num_positions = 7

# ----------------- ONE-HOT ENCODING -----------------
def encode_grana(grana):
    vec = []
    for val in grana:
        one_hot = [0]*num_nodes
        one_hot[val-1] = 1
        vec.extend(one_hot)
    return vec

X_train = np.array([encode_grana(row) for row in df.values])
y_train = df.values

# ----------------- TRAIN REGRESSOR -----------------
reg = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
reg.fit(X_train, y_train)

# ----------------- FUNCTION FOR BATCH PREDICTION -----------------
def process_batch(batch):
    X_batch = np.array([encode_grana(c) for c in batch])
    y_pred = reg.predict(X_batch)
    scores = y_pred.sum(axis=1)
    idx = np.argmax(scores)
    return (scores[idx], sorted(batch[idx]))

# ----------------- PARALLEL BATCH PREDICTION -----------------
all_nodes = list(range(1, num_nodes+1))
all_combs_iter = combinations(all_nodes, 7)
batch_size = 500000
max_score = -1
predicted_grana = None

def generate_batches(iterator, size):
    while True:
        batch = []
        try:
            for _ in range(size):
                batch.append(next(iterator))
        except StopIteration:
            pass
        if not batch:
            break
        yield batch

# ThreadPoolExecutor for parallel processing
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_batch, batch) for batch in generate_batches(all_combs_iter, batch_size)]
    for future in as_completed(futures):
        score, grana = future.result()
        if score > max_score:
            max_score = score
            predicted_grana = grana

print("Predicted next grana (7 nodes):", predicted_grana)
# Predicted next grana (7 nodes): [18, 29, x, y, z, 38, 39]

