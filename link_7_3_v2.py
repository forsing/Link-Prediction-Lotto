# LOTO 7/39 - paralelizovana batch predikcija sledeće grane

# RandomForestRegressor



# https://graphsinspace.net
# https://tigraphs.pmf.uns.ac.rs



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
from itertools import combinations, islice
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import os
import random

# ----------------- LOAD CSV FILE -----------------
SEED = 39
np.random.seed(SEED)
random.seed(SEED)

file_path = "/data/loto7_4586_k24.csv"
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

X_train = np.array([encode_grana(row) for row in df.values], dtype=np.float32)
y_train = df.values

# ----------------- TRAIN REGRESSOR -----------------
print("[start] Trening RandomForest modela...")
reg = RandomForestRegressor(n_estimators=220, random_state=SEED, n_jobs=-1)
reg.fit(X_train, y_train)
print("[ok] RandomForest istreniran. Kreće pretraga kombinacija...\n")

# ----------------- FUNCTION FOR BATCH PREDICTION -----------------
def process_batch(batch):
    X_batch = np.array([encode_grana(c) for c in batch], dtype=np.float32)
    y_pred = reg.predict(X_batch)
    scores = y_pred.sum(axis=1)
    idx = np.argmax(scores)
    return (scores[idx], sorted(batch[idx]))

def generate_batches(iterator, size):
    while True:
        batch = list(islice(iterator, size))
        if not batch:
            break
        yield batch

# ----------------- PARALLEL BATCH PREDICTION -----------------
all_nodes = list(range(1, num_nodes+1))
all_combs_iter = combinations(all_nodes, 7)
batch_size = 25000
max_score = -1
predicted_grana = None

# v2: ograničen broj aktivnih futures (bez gomilanja svih batch-eva u RAM)
max_workers = max(2, (os.cpu_count() or 4))
max_inflight = max_workers * 2
progress_every = 200

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    batch_iter = generate_batches(all_combs_iter, batch_size)
    inflight = set()
    processed_batches = 0

    for _ in range(max_inflight):
        try:
            batch = next(batch_iter)
        except StopIteration:
            break
        inflight.add(executor.submit(process_batch, batch))

    while inflight:
        done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
        for future in done:
            score, grana = future.result()
            processed_batches += 1
            if score > max_score:
                max_score = score
                predicted_grana = grana
            if processed_batches == 1:
                print(
                    f"[progress] first batch done "
                    f"best_score={max_score:.4f} best_grana={predicted_grana}"
                )
            if processed_batches % progress_every == 0:
                print(
                    f"[progress] batches={processed_batches} "
                    f"best_score={max_score:.4f} best_grana={predicted_grana}"
                )
            try:
                batch = next(batch_iter)
                inflight.add(executor.submit(process_batch, batch))
            except StopIteration:
                pass

print("Predicted next grana (7 nodes):", predicted_grana)



"""
isti princip kao ranije, ali bez gomilanja svih futures u RAM,
ograničen broj aktivnih poslova (max_inflight),
batch_size=25000,
float32 za brži/štedljiviji rad,
početni i progres ispis (start, first batch, pa na svakih 200 batch-eva),
"""


"""
[start] Trening RandomForest modela...
[ok] RandomForest istreniran. Kreće pretraga kombinacija...

[progress] first batch done best_score=135.6455 best_grana=[1, 2, 7, 14, x, y, z]
[progress] batches=200 best_score=186.3682 best_grana=[1, 26, 30, 31, x, y, z]
[progress] batches=400 best_score=196.8091 best_grana=[4, 25, 30, 33, x, y, z]
[progress] batches=600 best_score=218.0409 best_grana=[14, 29, 34, 36, x, y, z]

Predicted next grana (7 nodes): [18, 29, 34, 36, x, y, z]
"""
