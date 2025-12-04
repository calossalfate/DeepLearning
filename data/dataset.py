import pandas as pd

df = pd.read_csv("data/comentarios.csv", encoding="latin-1")
print(len(df))
print(df["sentimiento"].value_counts())
