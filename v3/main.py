# %%
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, cast
from tqdm import tqdm

try:
    from data.loader import load_dataframes
except ModuleNotFoundError:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from data.loader import load_dataframes

from models.embeddings.text_encoder import TextEncoder

# %%
data = load_dataframes(
    database_name="liebre_dev", business_id="bu-651"
)

# %%
to_embed = set()


def pattern_builder(data):
    for i, je in tqdm(data.journal_entry.iterrows()):
        pattern = ""
        entry_lines = data.entry_line[
            data.entry_line["journal_entry_id"] == je["journal_entry_id"]
        ]
        for i, el in entry_lines.iterrows():
            
            pattern += f"{data.ledger_account[data.ledger_account['ledger_account_id'] == el['ledger_account_id']]["number"].values}\n"
            pattern += f"{el['debit'] if el['debit'] != 0 else 'credit'}: {el['credit'] if el['debit'] == 0 else el['debit']}\n"
            
            # pattern.append({
            #     "account_id": data.ledger_account[data.ledger_account["ledger_account_id"] == el["ledger_account_id"]]["number"],
            #     "type": "debit" if el["debit"] == 0 else "credit",
            #     "amount": el["credit"] if el["debit"] == 0 else el["debit"]
            # })
        to_embed.add((str(je["journal_entry_id"]), je["description"], pattern))
    return to_embed


to_embed = pattern_builder(data)
  

# %%
data.ledger_account["number"]
# %%


import numpy as np
from sklearn.neighbors import NearestNeighbors

X = np.random.rand(100, 10)
q = np.random.rand(10)

# X: (n_samples, dim) embeddings, q: (dim,) query vector
nn = NearestNeighbors(n_neighbors=10, algorithm='auto', metric='minkowski', p=2)  # Euclidean
nn.fit(X)

dists, idxs = nn.kneighbors(q.reshape(1, -1), return_distance=True)  # shapes: (1,k), (1,k)
neighbors = [(int(i), float(d)) for i, d in zip(idxs[0], dists[0])]
print(neighbors)

#%%

arr = np.ones((1,50)) * 0.5

#%%
import matplotlib.pyplot as plt

def get_sin_wave(l,p):
    return np.sin(np.arange(l) / l * 2 * np.pi + ( p * 2*np.pi / 31))


plt.legend()
plt.show()