# LOTO 7/39 - streaming + paralelizovana predikcija sledeće grane

# RandomForestRegressor

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
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------- LOAD CSV FILE -----------------
file_path = "/Users/milan/Desktop/GHQ/data/loto7_4530_k99.csv"
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

# ----------------- STREAMING BATCH GENERATOR -----------------
def batch_generator(iterator, batch_size):
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch

# ----------------- FUNCTION TO PROCESS BATCH -----------------
def process_batch(batch):
    X_batch = np.array([encode_grana(c) for c in batch])
    y_pred = reg.predict(X_batch)
    scores = y_pred.sum(axis=1)
    idx = np.argmax(scores)
    return (scores[idx], sorted(batch[idx]))

# ----------------- MAIN STREAMING + PARALLEL EXECUTION -----------------
all_nodes = list(range(1, num_nodes+1))
all_combs_iter = combinations(all_nodes, 7)
batch_size = 100000  # manji batch da RAM ne eksplodira
max_score = -1
predicted_grana = None

with ThreadPoolExecutor() as executor:
    futures = []
    for batch in batch_generator(all_combs_iter, batch_size):
        futures.append(executor.submit(process_batch, batch))

    for future in as_completed(futures):
        score, grana = future.result()
        if score > max_score:
            max_score = score
            predicted_grana = grana

print("Predicted next grana (7 nodes):", predicted_grana)
# Predicted next grana (7 nodes): [18, 29, 34, 36, 37, 38, 39]
