# %%
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, cast, Optional
import pickle
from tqdm import tqdm

import pandas as pd

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
# my_df = pd.DataFrame()

# def pattern_builder(data):
#     global my_df
#     for i, je in tqdm(data.journal_entry[:100].iterrows()):
#         pattern = ""
#         entry_lines = data.entry_line[
#             data.entry_line["journal_entry_id"] == je["journal_entry_id"]
#         ]
#         for j, el in entry_lines.iterrows():
            
#             pattern += f"{data.ledger_account[data.ledger_account['ledger_account_id'] == el['ledger_account_id']]["number"].values} -> "
#             pattern += f"{'debit' if el['debit'] != 0 else 'credit'}\n"
        
#         my_df = pd.concat([my_df, pd.DataFrame({
#             "journal_entry_id": [je["journal_entry_id"]],
#             "description": [je["description"]],
#             "pattern": [pattern]
#         })], ignore_index=True)

#%%
data.journal_entry

#%%
def pattern_builder_fast(data):
    # Step 1: Get first 100 journal entries
    journal_subset = data.journal_entry[['journal_entry_id', 'description']].copy()
    print(f"Journal subset columns: {journal_subset.columns.tolist()}")
    print(f"Journal subset shape: {journal_subset.shape}")
    
    # Step 2: Merge entry lines with ledger accounts
    entry_with_accounts = data.entry_line.merge(
        data.ledger_account[['ledger_account_id', 'number']], 
        on='ledger_account_id', 
        how='left'
    )
    print(f"Entry with accounts columns: {entry_with_accounts.columns.tolist()}")
    
    # Step 3: Filter to only the 100 journal entries we care about
    entry_with_accounts = entry_with_accounts[
        entry_with_accounts['journal_entry_id'].isin(journal_subset['journal_entry_id'])
    ]
    
    # Step 4: Create pattern lines
    entry_with_accounts['pattern_line'] = (
        entry_with_accounts['number'].astype(str) + ' -> ' + 
        entry_with_accounts['debit'].apply(lambda x: 'debit' if x != 0 else 'credit')
    )
    
    # Step 5: Aggregate patterns by journal_entry_id
    patterns = (entry_with_accounts
                .groupby('journal_entry_id')['pattern_line']
                .apply('\n'.join)
                .reset_index()
                .rename(columns={'pattern_line': 'pattern'}))
    
    print(f"Patterns columns: {patterns.columns.tolist()}")
    
    # Step 6: Merge with journal entries to get descriptions
    result_df = journal_subset.merge(patterns, on='journal_entry_id', how='left')
    
    print(f"Result columns: {result_df.columns.tolist()}")
    
    return result_df[['journal_entry_id', 'description', 'pattern']]



my_df = pattern_builder_fast(data)

#%%

desc_to_pattern = defaultdict(list)

for desc in my_df["description"].unique():
    counts = my_df[my_df["description"] == desc]["pattern"].value_counts()
    for value, count in counts.items():
        desc_to_pattern[desc].append((value, count))

desc_to_pattern


#%%

# arr = np.ones((1,50)) * 0.5

# #%%
# import matplotlib.pyplot as plt

# def get_sin_wave(l,p):
#     return np.sin(np.arange(l) / l * 2 * np.pi + ( p * 2*np.pi / 31))


# plt.legend()
# plt.show()

#%%
import torch
import torch.nn.functional as F

class TextKNN:
    def __init__(self, texts: list[str], objects: list[Any], metric="cosine", device="cpu"):
        """
        embeddings: torch.Tensor [N, D]
        objects: list of length N
        metric: "cosine" or "l2"
        """
        self.texts = list(texts)
        self.objects = list(objects)
        self.metric = metric
        self.device = device
        self.encoder = TextEncoder()
        embs, stats = self.encoder.encode(texts)
        self.embeddings = embs.to(device)
        self.stats = stats

    @torch.no_grad()
    def query(self, text, k=5):
        # Get embedding for the query using TextEncoder
        q, stats = self.encoder.encode(text)

        if not isinstance(q, torch.Tensor):
            q = torch.tensor(q, dtype=torch.float32)

        q = q.to(self.device)
        if q.ndim == 1:
            q = q.unsqueeze(0)

        if self.metric == "cosine":
            q = F.normalize(q, p=2, dim=-1)
            # similarity = dot product
            sims = torch.matmul(self.embeddings, q.T).squeeze(-1)  # [N]
            # take top-k highest similarity
            vals, idx = torch.topk(sims, k=k, largest=True)

            # convert similarities to cosine distances (optional)
            distances = 1 - vals.cpu().numpy()

        else:  # L2 distance
            diff = self.embeddings - q
            dists = torch.sum(diff * diff, dim=1)  # squared L2 distance
            vals, idx = torch.topk(-dists, k=k)  # negative for smallest distances
            distances = (-vals).cpu().numpy()

        neighbors = [self.objects[i] for i in idx.cpu().numpy()]
        return neighbors, distances

class PersistentTextKNN:
    """
    Wrapper class that handles persistence for TextKNN.
    
    Usage:
        # Create and save
        knn = PersistentTextKNN.create(texts, objects, save_path="my_knn")
        
        # Load and query
        knn = PersistentTextKNN.load("my_knn")
        results, distances = knn.query("search text", k=5)
    """
    
    def __init__(self, knn: TextKNN, save_path: Optional[str] = None):
        self.knn = knn
        self.save_path = save_path
    
    @classmethod
    def create(cls, texts: list[str], objects: list[Any], 
               metric: str = "cosine", device: str = "cpu",
               save_path: Optional[str] = None):
        """
        Create a new TextKNN and optionally save it immediately.
        
        Args:
            texts: List of text strings
            objects: List of objects corresponding to texts
            metric: "cosine" or "l2"
            device: PyTorch device
            save_path: Optional path to save the KNN (without extension)
        """
        knn = TextKNN(texts, objects, metric=metric, device=device)
        instance = cls(knn, save_path)
        
        if save_path:
            instance.save(save_path)
        
        return instance
    
    def save(self, save_path: Optional[str] = None):
        """
        Save the TextKNN to disk.
        
        Args:
            save_path: Directory/file path (without extension). 
                      If None, uses self.save_path
        """
        path = save_path or self.save_path
        if path is None:
            raise ValueError("No save path provided")
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings as torch tensor
        torch.save(self.knn.embeddings, path / "embeddings.pt")
        
        # Save metadata as pickle
        metadata = {
            'texts': self.knn.texts,
            'objects': self.knn.objects,
            'metric': self.knn.metric,
            'device': self.knn.device,
            'stats': self.knn.stats
        }
        with open(path / "metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        self.save_path = str(path)
        print(f"TextKNN saved to {path}")
    
    @classmethod
    def load(cls, load_path: str, device: Optional[str] = None):
        path = Path(load_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist")
        
        # Load metadata
        with open(path / "metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        # Use provided device or fall back to saved device
        target_device = device or metadata['device']
        
        # Create empty TextKNN object (we'll populate it manually)
        knn = object.__new__(TextKNN)
        knn.texts = metadata['texts']
        knn.objects = metadata['objects']
        knn.metric = metadata['metric']
        knn.device = target_device
        knn.stats = metadata['stats']
        knn.encoder = TextEncoder()  # Recreate encoder
        
        # Load embeddings
        knn.embeddings = torch.load(path / "embeddings.pt", map_location=target_device)
        
        print(f"TextKNN loaded from {path}")
        return cls(knn, str(path))
    
    def query(self, text: str, k: int = 5):
        return self.knn.query(text, k=k)
    
    def __len__(self):
        return len(self.knn.texts)
    
    def __repr__(self):
        return (f"PersistentTextKNN(n_items={len(self)}, metric='{self.knn.metric}', "
                f"device='{self.knn.device}', save_path='{self.save_path}')")

kNN = PersistentTextKNN.create(
    texts=desc_to_pattern.keys(),
    objects=desc_to_pattern.values(),
    save_path="./knn",
    metric="cosine",
)

#%%
kNN.query("sale of goods")

# %%