import pandas as pd
import time

def read_matrix(name, file, lineterminator=None):
    print(f"--- Reading {name} matrix ---")
    start = time.time()
    
    print("Loading...")
    matrix = pd.read_csv(f"../data/{file}.csv", lineterminator=lineterminator)
    bef = len(matrix)
    
    print("Cleaning...")
    matrix = matrix.dropna().drop_duplicates()
    
    print(f"{100 * (1 - len(matrix) / bef):.2f}% of the data has been filtered out")

    print(f"--- Readed in {time.time() - start:.4f}s ---")

    return matrix
