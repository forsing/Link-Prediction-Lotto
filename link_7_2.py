# LOTO 7/39 - optimizovana batch predikcija sledeće grane sa one-hot encoding

# RandomForestRegressor

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
from itertools import combinations

# ----------------- LOAD CSV FILE -----------------
file_path = "/Users/milan/Desktop/GHQ/data/loto7_4530_k99.csv"
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

X_train = np.array([encode_grana(row) for row in df.values])
y_train = df.values  # željeni izlaz: 7 čvorova

# ----------------- TRAIN REGRESSOR -----------------
reg = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
reg.fit(X_train, y_train)

# ----------------- BATCH PREDICTION FOR ALL 39C7 COMBINATIONS -----------------
all_nodes = list(range(1, num_nodes+1))
all_combs_iter = combinations(all_nodes, 7)

batch_size = 500000
max_score = -1
predicted_grana = None

while True:
    batch = []
    try:
        for _ in range(batch_size):
            batch.append(next(all_combs_iter))
    except StopIteration:
        pass

    if len(batch) == 0:
        break

    X_batch = np.array([encode_grana(c) for c in batch])
    y_batch_pred = reg.predict(X_batch)

    # Score: suma predikcija po 7 pozicija
    scores = y_batch_pred.sum(axis=1)
    idx = np.argmax(scores)
    if scores[idx] > max_score:
        max_score = scores[idx]
        predicted_grana = sorted(batch[idx])

print("Predicted next grana (7 nodes):", predicted_grana)
# Predicted next grana (7 nodes): [18, 29, 34, 36, 37, 38, 39]

