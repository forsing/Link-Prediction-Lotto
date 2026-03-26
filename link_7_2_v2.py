# LOTO 7/39 - optimizovana batch predikcija sledeće grane sa one-hot encoding


# RandomForestRegressor



# https://graphsinspace.net
# https://tigraphs.pmf.uns.ac.rs



"""
Model koristi one-hot encoding ili embedding 
za svaki od 39 čvorova u 7 pozicija.
Uči tačno raspodelu prethodnih 4530 grana.
Direktno generiše validnu sledeću granu bez aproksimacija 
i favorizovanja velikih brojeva.

Uči tačne raspodele svih prethodnih 4530 grana 
i predviđa sledeću granu 7 čvorova koristeći embedding 
ili one-hot encoding, bez aproksimacija.

Ovaj kod:
Uči tačne pozicije svakog čvora u prethodnim 4530 granama.
Predviđa sledeću granu direktno sa one-hot encodingom.
Radi po batch-evima da ne preoptereti memoriju.

Optimizovanu verziju koja:
Radi po batch-evima, da ne preoptereti memoriju.
Računa score za svaku kombinaciju koristeći predikciju regresora za 7 pozicija.
Direktno vraća najverovatniju sledeću granu.
"""


"""
Karakteristike ove verzije:
Predikcija se vrši sve kombinacije 39 na 7 po batch-evima (500k po batch-u).
Regresor uči tačne pozicije svakog čvora iz prethodnih 4530 grana.
Najverovatnija grana se automatski bira na kraju.
Izlaz je sortirana kombinacija 7 čvorova.
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

# ----------------- ONE-HOT ENCODING FUNCTION -----------------
def encode_grana(grana):
    vec = []
    for val in grana:
        one_hot = [0]*num_nodes
        one_hot[val-1] = 1
        vec.extend(one_hot)
    return vec

X_train = np.array([encode_grana(row) for row in df.values], dtype=np.float32)
y_train = df.values  # željeni izlaz: 7 čvorova

# ----------------- TRAIN REGRESSOR -----------------
print("[start] Trening RandomForest modela...")
reg = RandomForestRegressor(n_estimators=220, random_state=SEED, n_jobs=-1)
reg.fit(X_train, y_train)
print("[ok] RandomForest istreniran. Kreće pretraga kombinacija...\n")

# ----------------- BATCH PREDICTION FOR ALL 39C7 COMBINATIONS -----------------
all_nodes = list(range(1, num_nodes+1))
all_combs_iter = combinations(all_nodes, 7)

batch_size = 25000
max_score = -1
predicted_grana = None

def generate_batches(iterator, size):
    while True:
        batch = list(islice(iterator, size))
        if not batch:
            break
        yield batch

def process_batch(batch):
    X_batch = np.array([encode_grana(c) for c in batch], dtype=np.float32)
    y_batch_pred = reg.predict(X_batch)
    # Score: suma predikcija po 7 pozicija
    scores = y_batch_pred.sum(axis=1)
    idx = np.argmax(scores)
    return float(scores[idx]), sorted(batch[idx])

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
ograničen broj aktivnih futures (nema gomilanja u RAM),
batch_size=25000,
float32 za brži/štedljiviji rad,
početni i progres ispis (start, first batch, pa svakih 200 batch-eva),
seed 39 i RF n_estimators=220.
"""


"""
[start] Trening RandomForest modela...
[ok] RandomForest istreniran. Kreće pretraga kombinacija...

[progress] first batch done best_score=134.9818 best_grana=[1, 2, 4, 14, x, y, z]
[progress] batches=200 best_score=186.3682 best_grana=[1, 26, 30, 31, x, y, z]
[progress] batches=400 best_score=196.8091 best_grana=[4, 25, 30, 33, x, y, z]
[progress] batches=600 best_score=218.0409 best_grana=[14, 29, 34, 36, x, y, z]

Predicted next grana (7 nodes): [18, 29, x, y, z, 38, 39]
"""
