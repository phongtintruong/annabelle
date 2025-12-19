from typing import List, Dict, Any
import pandas as pd
from pathlib import Path
import yaml
import torch 
from tqdm import tqdm
from openai import OpenAI, APIConnectionError
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

import config

class EmbeddingManager(ABC): 
    @abstractmethod
    def __init__(self, base_url: str | None = None, model: str | None = None, timeout: float | None = None):
        pass 

    @abstractmethod
    def encode(self, dialog: List[str]) -> List[List[float]]:
        pass

    def score_similarity(self, text1: str, text2: str) -> float:
        pass


class Qwen3Embedding(EmbeddingManager):
    """
    Manager for Qwen3 embedding model served by SGLang.
    """
    def __init__(self, base_url: str | None = None, model: str | None = None, timeout: float | None = None):
        self.base_url = base_url or config.EMBEDDING_BASE_URL
        self.model = model or config.EMBEDDING_MODEL
        self.timeout = timeout or config.EMBEDDING_TIMEOUT
        self.encoding_format = config.EMBEDDING_ENCODING_FORMAT
        
        self.client = OpenAI(
            base_url=self.base_url, 
            api_key=config.OPENAI_API_KEY, 
            timeout=self.timeout,
            max_retries=2 
        )

    def _post_embeddings_batch(self, texts: List[str]) -> List[Dict]:
        """
        Hàm gửi request cho 1 batch.
        Trả về list object embedding kèm index để sort lại sau.
        """
        try:
            resp = self.client.embeddings.create(
                model=self.model,
                input=texts,
                encoding_format=self.encoding_format,
            )
            return resp.data
        except APIConnectionError as exc:
            print(f"Connection error: {exc}")
            raise exc
        except Exception as exc:
            print(f"Error processing batch: {exc}")
            raise exc

    def encode(self, dialog: List[str], batch_size: int = 64, max_workers: int = 8) -> List[List[float]]:
        if not dialog:
            return []

        chunks = [dialog[i:i + batch_size] for i in range(0, len(dialog), batch_size)]
        
        all_embeddings = [None] * len(dialog)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch_info = {
                executor.submit(self._post_embeddings_batch, chunk): (i * batch_size, chunk) 
                for i, chunk in enumerate(chunks)
            }

            for future in tqdm(as_completed(future_to_batch_info), total=len(chunks), desc="Embedding Progress"):
                start_index, chunk = future_to_batch_info[future]
                try:
                    data = future.result()
                    for item in data:
                        # item.index là index cục bộ trong batch
                        global_idx = start_index + item.index
                        all_embeddings[global_idx] = item.embedding
                except Exception as exc:
                    print(f"Batch starting at index {start_index} failed: {exc}")
                    raise exc

        if any(v is None for v in all_embeddings):
             raise ValueError("Some embeddings failed to be retrieved.")

        return all_embeddings

    def score_similarity(self, text1: str, text2: str) -> float:
        vecs = self.encode([text1, text2], batch_size=2, max_workers=1)
        v1 = torch.tensor(vecs[0])
        v2 = torch.tensor(vecs[1])
        return torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()

if __name__ == "__main__":
    embedder = Qwen3Embedding()
    
    data = ["Xin chào", "Hôm nay trời đẹp Hôm nay trời đẹp Hôm nay trời đẹp Hôm nay trời đẹpHôm nay trời đẹp Hôm nay trời đẹp Hôm nay trời đẹp Hôm nay trời đẹp Hôm nay trời đẹp Hôm nay trời đẹp Hôm nay trời đẹp Hôm nay trời đẹp Hôm nay trời đẹp Hôm nay trời đẹp Hôm nay trời đẹp Hôm nay trời đẹp Hôm nay trời đẹp Hôm nay trời đẹp Hôm nay trời đẹp Hôm nay trời đẹp Hôm nay trời đẹp Hôm nay trời đẹp Hôm nay trời đẹp Hôm nay trời đẹp Hôm nay trời đẹp Hôm nay trời đẹp Hôm nay trời đẹp Hôm nay trời đẹp Hôm nay trời đẹp Hôm nay trời đẹp Hôm nay trời đẹp Hôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹpHôm nay trời đẹp", "SGLang rất nhanh"] * 1000 
    
    vectors = embedder.encode(data, batch_size=32, max_workers=16)
    
    print(f"Got {len(vectors)} vectors.")
    print(f"Dimension: {len(vectors[0])}")