"""
Preparación y limpieza de datos
Entrada : politicES_phase_2_train_public.csv
Salida  : data/train_clean.csv, data/train_dev.csv, data/test_holdout.csv
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
RUTA_ENTRADA = "politicES_phase_2_train_public.csv"
os.makedirs("data", exist_ok=True)

# Actualizamos el nombre de la columna label a cluster_id
df = pd.read_csv(RUTA_ENTRADA)
df = df.rename(columns={"label": "cluster_id"})
print("Filas originales:", len(df))

# Eliminamos tweets duplicados
print("Duplicados eliminados:", df.tweet.duplicated().sum())
df = df.drop_duplicates(subset="tweet", keep="first").reset_index(drop=True)

# Añadimos una columna que le da un id incremental a cada tweet
df.insert(0, "row_id", np.arange(len(df)))

# Concatenamos en una nueva columna, strat, la ideología binaria y la profesión
df["strat"] = df.ideology_binary + "_" + df.profession

# Hacemos clusters stratificados, dividiendo en 6 folds y tomando tan solo uno para test
sgk6 = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=SEED)
_, test_idx = next(sgk6.split(df, df.strat, groups=df.cluster_id))
df["split"] = "dev"                # Aquí estamos marcando qué datos son de train
df.loc[test_idx, "split"] = "test" # y cuales de test

# Creamos 5 folds de entrenamiento, marcando en la columna fold a cual pertenece cada dato
df["fold"] = -1
dev = df[df.split == "dev"]
sgk5 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
for k, (_, va) in enumerate(sgk5.split(dev, dev.strat, groups=dev.cluster_id)):
    df.loc[dev.index[va], "fold"] = k
df.loc[df.split == "test", "fold"] = 5   # Marcamos la columna fold a 5 para test

# Preparamos la estructura de columnas de los 3 ficheros
cols = ["row_id", "cluster_id", "fold", "tweet", "gender",
        "profession", "ideology_binary", "ideology_multiclass"]
df[["row_id", "cluster_id", "split"] + cols[2:]].to_csv("data/train_clean.csv", index=False)
df[df.split == "dev"][cols].to_csv("data/train_dev.csv", index=False)
df[df.split == "test"][cols].drop(columns="fold").to_csv("data/test_holdout.csv", index=False)

# Comprobaciones finales para evitar data leakages o ids no unívocos
assert df.row_id.is_unique
assert set(df[df.split == "dev"].cluster_id).isdisjoint(df[df.split == "test"].cluster_id)
assert (df.groupby("cluster_id").fold.nunique() == 1).all()
print(df.groupby("split").agg(tweets=("row_id", "size"), clusters=("cluster_id", "nunique")))
print(df[df.split == "dev"].groupby("fold").agg(tweets=("row_id", "size"), clusters=("cluster_id", "nunique")))