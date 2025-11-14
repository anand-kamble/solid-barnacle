#%%
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from data.loader import load_dataframes
from models.text_encoder import OpenAIEmbeddingEncoder

#%%
data = load_dataframes(
    database_name="liebre_dev",
    max_connections=5,
    business_id="bu-651"
)

print("Loaded data:")
print(f"  Journal entries: {data.journal_entry.shape}")
print(f"  Entry lines: {data.entry_line.shape}")
print(f"  Ledger accounts: {data.ledger_account.shape}")

#%%
sample_descriptions = data.journal_entry["description"].dropna().head(10).tolist()

print(f"\nSample descriptions to embed ({len(sample_descriptions)}):")
for i, desc in enumerate(sample_descriptions, 1):
    print(f"  {i}. {desc[:80]}...")

#%%
encoder = OpenAIEmbeddingEncoder(
    model="text-embedding-ada-002",
    batch_size=10,
    device=torch.device("cpu"),
)

print("Encoder initialized")
print(f"  Model: {encoder.model}")
print(f"  Batch size: {encoder.batch_size}")
print(f"  Device: {encoder.device}")
print(f"  Cache size: {encoder.get_cache_size()}")

#%%
result = encoder.encode(sample_descriptions, return_cache_stats=True)
embeddings, stats = result

print("\nEmbedding results:")
print(f"  Embedding shape: {embeddings.shape}")
print(f"  Embedding dtype: {embeddings.dtype}")
print("\nCache statistics:")
if isinstance(stats, dict):
    for key, value in stats.items():
        print(f"  {key}: {value}")

#%%
print("\nEmbedding tensor info:")
print(f"  Shape: {embeddings.shape}")
print(f"  Mean: {embeddings.mean().item():.6f}")
print(f"  Std: {embeddings.std().item():.6f}")
print(f"  Min: {embeddings.min().item():.6f}")
print(f"  Max: {embeddings.max().item():.6f}")

#%%
duplicate_test = sample_descriptions[:3] + sample_descriptions[:2]

print("\nTesting duplicate handling:")
print(f"  Input descriptions: {len(duplicate_test)}")
print(f"  Unique descriptions: {len(set(duplicate_test))}")

result_dup = encoder.encode(duplicate_test, return_cache_stats=True)
embeddings_dup, stats_dup = result_dup

print("\nDuplicate test results:")
print(f"  Embedding shape: {embeddings_dup.shape}")
print("  Cache statistics:")
if isinstance(stats_dup, dict):
    for key, value in stats_dup.items():
        print(f"    {key}: {value}")

#%%
print(embeddings.mean().item())
print(embeddings_dup.mean().item())