import os
import torch
import numpy as np
import pandas as pd
import faiss
import pickle
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from custom_llm_model import QwenForNextMessagePrediction
import config

# --- CONFIG ---
CHECKPOINT_PATH = "checkpoints/epoch-1"
CORPUS_DATASET_ID = config.CORPUS_REPO_ID 
FAISS_INDEX_FILE = "test/corpus.index"      
TEXT_DB_FILE = "test/corpus.pkl"     
TOP_K = 5
# --------------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model_wrapper = QwenForNextMessagePrediction(
    model_name=config.MODEL_NAME,
    output_embedding_size=config.TARGET_EMBEDDING_SIZE,
    quantization_config=bnb_config,
    attn_implementation="flash_attention_2" if config.USE_FLASH_ATTENTION else "eager"
)

model_wrapper.qwen = PeftModel.from_pretrained(
    model_wrapper.qwen,
    CHECKPOINT_PATH,
    is_trainable=False
)

head_path = os.path.join(CHECKPOINT_PATH, "head.pt")
if os.path.exists(head_path):
    head_state_dict = torch.load(head_path, map_location="cpu")
    model_wrapper.projection_head.load_state_dict(head_state_dict)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_wrapper.projection_head.to(device)
model_wrapper.eval()



index = None
corpus_texts = []


if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(TEXT_DB_FILE):
    index = faiss.read_index(FAISS_INDEX_FILE)

    with open(TEXT_DB_FILE, "rb") as f:
        corpus_texts = pickle.load(f)
        
    print(f"✓ Database loaded. Total items: {len(corpus_texts)}")

else:
    print("Local database not found. Downloading dataset to build...")
    dataset = load_dataset(CORPUS_DATASET_ID, split="train", token=config.HF_TOKEN)
    
    print("Building Text Database...")
    corpus_texts = dataset["response"]
    
    print("Building FAISS index...")
    corpus_embeddings = np.array(dataset["embedding"], dtype=np.float32)
    faiss.normalize_L2(corpus_embeddings)

    index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
    index.add(corpus_embeddings)
    
    index_dir = os.path.dirname(FAISS_INDEX_FILE)
    if index_dir:
        os.makedirs(index_dir, exist_ok=True)
        
    text_db_dir = os.path.dirname(TEXT_DB_FILE)
    if text_db_dir:
        os.makedirs(text_db_dir, exist_ok=True)

    faiss.write_index(index, FAISS_INDEX_FILE)
    
    with open(TEXT_DB_FILE, "wb") as f:
        pickle.dump(corpus_texts, f)
        
    del dataset 

# ----------------------------------------

def get_embedding(text):
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=config.MAX_SEQ_LENGTH,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        embedding = model_wrapper(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
    return embedding

def normalize_embedding(embedding):
    """
    CHUẨN HÓA L2 GiỐNG TRAIN
    Input: embedding tensor [B, D] hay [1, D]
    Output: normalized embedding [B, D]
    """
    embedding = embedding.float()
    
    norm = torch.norm(embedding, p=2, dim=1, keepdim=True)
    normalized = embedding / (norm + 1e-8)
    
    return normalized

def search_response(history_text, k=5):
    query_vec = get_embedding(history_text)
    query_vec_normalized = normalize_embedding(query_vec)
    query_vec_np = query_vec_normalized.float().cpu().numpy()
    
    distances, indices = index.search(query_vec_np, k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        text = corpus_texts[int(idx)]
        score = distances[0][i]  
        results.append({"text": text, "score": float(score)})
        
    return results

if __name__ == "__main__":
    df = pd.read_csv("test/test_data.csv")
    for i, row in enumerate(df.itertuples()):
        history = row.anchor
        response = row.positive    
        
        # print(f"\nQuery {i}: {history}")
        print(f"Expected Response: {response}\n")
        
        hits = search_response(history, k=TOP_K)
        
        for i, hit in enumerate(hits):
            print(f"Top {i+1} (Score: {hit['score']:.4f}):\n{hit['text']}\n")

        print("--------------------------------------------------")