# LOTO 7/39 - streaming + paralelizovana predikcija sledeće grane


# RandomForestRegressor



# https://graphsinspace.net
# https://tigraphs.pmf.uns.ac.rs



"""
Još brža verziju sa većom paralelizacijom 
i manje memorijsko-intenzivnim pristupom, 
koristeći generator + streaming score ažuriranje, 
bez da ikada učitavamo batch ceo u RAM.
"""


"""
Maksimalno optimizovane verzije:
Ne učitava celu batch listu u RAM, već koristi generator + streaming.
Ažurira maksimum u realnom vremenu dok iterira kroz sve kombinacije.
Paralelizacija je po chunk-evima, tako da RAM ne raste 
i sve 15,380,937 kombinacija se obrađuju efikasno.
"""


"""
Ključne karakteristike ove verzije:

Streaming generacija kombinacija 
– nikada ne držimo celu listu 15 miliona kombinacija u memoriji.

Paralelizacija po batch-evima 
– koristi sva dostupna CPU jezgra.

Ažuriranje maksimuma u realnom vremenu 
– vraća samo najverovatniju granu.

Full compliance sa zadatkom 
– 39 čvorova, 7 po grani, koristi Random Forest Regressor 
  i uči iz svih prethodnih 4530 grana.

Ovo je finalna, maksimalno optimizovana verzija
"""



# LOTO 7/39 - streaming + paralelizovana predikcija sledeće grane

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

# ----------------- STREAMING BATCH GENERATOR -----------------
def batch_generator(iterator, batch_size):
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch

# ----------------- FUNCTION TO PROCESS BATCH -----------------
def process_batch(batch):
    X_batch = np.array([encode_grana(c) for c in batch], dtype=np.float32)
    y_pred = reg.predict(X_batch)
    scores = y_pred.sum(axis=1)
    idx = np.argmax(scores)
    return (scores[idx], sorted(batch[idx]))

# ----------------- MAIN STREAMING + PARALLEL EXECUTION -----------------
all_nodes = list(range(1, num_nodes+1))
all_combs_iter = combinations(all_nodes, 7)
batch_size = 25000  # v2: manji batch + ograničen broj aktivnih futures
max_score = -1
predicted_grana = None

# v2: ne čuvamo sve futures u listi (to je bio najveći RAM problem),
# već držimo samo mali broj aktivnih poslova.
max_workers = max(2, (os.cpu_count() or 4))
max_inflight = max_workers * 2
progress_every = 200

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    batch_iter = batch_generator(all_combs_iter, batch_size)
    inflight = set()
    processed_batches = 0

    # initial fill
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

            # submit next batch immediately (streaming)
            try:
                batch = next(batch_iter)
                inflight.add(executor.submit(process_batch, batch))
            except StopIteration:
                pass

print("Predicted next grana (7 nodes):", predicted_grana)




"""
Glavno v2 poboljšanje:

rešeno usko grlo memorije: više se ne čuvaju svi futures odjednom, nego radi sa ograničenim brojem aktivnih poslova (max_inflight), pa je stvarno streaming.
Dodatno:

SEED=39,
float32 za X_train i batch-eve,
nešto jači RandomForestRegressor (n_estimators=220),
manji batch_size=25000 radi stabilnijeg RAM ponašanja.




[start] Trening RandomForest modela...
[ok] RandomForest istreniran. Kreće pretraga kombinacija...
posle prvog batch-a: [progress] first batch done ...

ispisuje progres na svakih 200 batch-eva:
broj obrađenih batch-eva,
trenutni best_score,
trenutnu najbolju granu.
gušći ispis, progress_every = 200 na npr. 50.

"""




"""
[start] Trening RandomForest modela...
[ok] RandomForest istreniran. Kreće pretraga kombinacija...

[progress] first batch done best_score=134.9818 best_grana=[x, y, z, 14, 29, 37, 38]
[progress] batches=200 best_score=186.3682 best_grana=[x, y, z, 31, 35, 38, 39]
[progress] batches=400 best_score=196.8091 best_grana=[x, y, z, 33, 34, 35, 38]
[progress] batches=600 best_score=218.0409 best_grana=[x, y, z, 36, 37, 38, 39]

Predicted next grana (7 nodes): [x, y, z, 36, 37, 38, 39]

"""
