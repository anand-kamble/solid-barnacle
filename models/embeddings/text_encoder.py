import hashlib
import pickle
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn as nn
from openai import OpenAI


class TextEncoder(nn.Module):
    def __init__(
        self,
        model: str = "text-embedding-ada-002",
        cache_dir: Optional[Union[str, Path]] = None,
        batch_size: int = 100,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.device = device or torch.device("cpu")
        
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "openai_embeddings"
        else:
            cache_dir = Path(cache_dir)
        
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = cache_dir / f"embeddings_{model.replace('-', '_')}.pkl"
        
        self.cache: dict[str, torch.Tensor] = self._load_cache()
        self.client = OpenAI()
        
    def _load_cache(self) -> dict[str, torch.Tensor]:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "rb") as f:
                    cache = pickle.load(f)
                    for key, value in cache.items():
                        if isinstance(value, list):
                            cache[key] = torch.tensor(value, dtype=torch.float32)
                    return cache
            except Exception:
                return {}
        return {}
    
    def _save_cache(self):
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.cache, f)
        except Exception:
            pass
    
    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    
    def _fetch_embeddings(self, texts: List[str]) -> List[torch.Tensor]:
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        embeddings = []
        for item in response.data:
            emb = torch.tensor(item.embedding, dtype=torch.float32)
            embeddings.append(emb)
        return embeddings
    
    def forward(
        self,
        descriptions: Union[str, List[str]],
        return_cache_stats: bool = True,
    ) -> tuple[torch.Tensor, dict]:
        if isinstance(descriptions, str):
            descriptions = [descriptions]
        
        unique_descriptions = {}
        description_to_idx = []
        
        for idx, desc in enumerate(descriptions):
            desc_hash = self._hash_text(desc)
            if desc_hash not in unique_descriptions:
                unique_descriptions[desc_hash] = desc
            description_to_idx.append(desc_hash)
        
        embeddings_dict = {}
        texts_to_fetch = []
        hashes_to_fetch = []
        cache_hits_count = 0
        cache_before_fetch = set(self.cache.keys())
        
        for desc_hash, desc in unique_descriptions.items():
            if desc_hash in self.cache:
                embeddings_dict[desc_hash] = self.cache[desc_hash]
                cache_hits_count += 1
            else:
                texts_to_fetch.append(desc)
                hashes_to_fetch.append(desc_hash)
        
        if texts_to_fetch:
            for i in range(0, len(texts_to_fetch), self.batch_size):
                batch_texts = texts_to_fetch[i : i + self.batch_size]
                batch_hashes = hashes_to_fetch[i : i + self.batch_size]
                
                batch_embeddings = self._fetch_embeddings(batch_texts)
                
                for hash_key, emb in zip(batch_hashes, batch_embeddings):
                    embeddings_dict[hash_key] = emb
                    self.cache[hash_key] = emb
            
            self._save_cache()
        
        result_embeddings = [
            embeddings_dict[desc_hash].to(self.device)
            for desc_hash in description_to_idx
        ]
        
        output = torch.stack(result_embeddings)
        
        if return_cache_stats:
            total_cache_hits = sum(
                1 for desc_hash in description_to_idx
                if desc_hash in cache_before_fetch
            )
            stats = {
                "total": len(descriptions),
                "cache_hits": total_cache_hits,
                "cache_misses": len(descriptions) - total_cache_hits,
                "unique_descriptions": len(unique_descriptions),
                "unique_cache_hits": cache_hits_count,
                "unique_cache_misses": len(texts_to_fetch),
                "api_calls": len(texts_to_fetch),
            }
            return output, stats
        
        return output, {}
    
    def encode(
        self,
        descriptions: Union[str, List[str]],
        return_cache_stats: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict]]:
        return self.forward(descriptions, return_cache_stats)
    
    def clear_cache(self):
        self.cache.clear()
        if self.cache_file.exists():
            self.cache_file.unlink()
    
    def get_cache_size(self) -> int:
        return len(self.cache)

