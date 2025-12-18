import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import logging
import numpy as np
import argparse
from tqdm import tqdm
import torch 
import faiss
from transformers import AutoTokenizer
from datasets import load_dataset
from qwen import Qwen3Embedding
import json

def setup_logging(log_file: str, level: int = logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

def read_dialog(data):
    """Parse format hội thoại từ dataset"""
    results = []
    dialog_list = data if isinstance(data, list) else json.loads(data)
    
    for item in dialog_list:
        role_raw = item.get("role", "unknown")
        turn = "OP" if role_raw == "assistant" else "CU"
        text = item.get("content", "").strip()
        if not text: continue   
        
        if results and results[-1]["role"] == turn:
            results[-1]["text"] += "\n" + text
        else:
            results.append({"role": turn, "text": text})
    return results

def format_history(turns, tokenizer):
    messages = []
    for t in turns:
        role = "assistant" if t["role"] == "OP" else "user"
        messages.append({"role": role, "content": t["text"]})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def count_tokens(text, tokenizer):
    """Đếm số token của text"""
    tokens = tokenizer.encode(text, add_special_tokens=True)
    return len(tokens)

def encode_in_batches(embedding_model, texts, batch_size=32, max_workers=8):
    """
    Encode texts thành embeddings với multithreading + chuẩn hóa L2
    
    Args:
        embedding_model: Model embedding (hỗ trợ multithreading)
        texts: List các text cần encode
        batch_size: Batch size cho mỗi request
        max_workers: Số threads song song
        
    Returns:
        np.ndarray: Embeddings đã chuẩn hóa L2, shape (n, d), dtype float32
    """
    total = len(texts)
    logging.info(f"Encoding {total} texts (batch_size={batch_size}, max_workers={max_workers})...")
    
    try:
        # Encode với multithreading
        all_embeddings = embedding_model.encode(
            texts,
            batch_size=batch_size,
            max_workers=max_workers,
        )
        
        # Convert to numpy float32
        if isinstance(all_embeddings, list):
            embeddings_np = np.array(all_embeddings, dtype=np.float32)
        else:
            embeddings_np = np.asarray(all_embeddings, dtype=np.float32)
        
        # Validate shape
        if embeddings_np.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got shape {embeddings_np.shape}")
        
        if embeddings_np.shape[0] != total:
            raise ValueError(f"Expected {total} embeddings, got {embeddings_np.shape[0]}")
        
        logging.info(f"Encoded shape: {embeddings_np.shape}, dtype: {embeddings_np.dtype}")
        
        # Normalize L2
        faiss.normalize_L2(embeddings_np)
        logging.info("✓ Normalized L2")
        
        return embeddings_np
        
    except Exception as e:
        logging.error(f"Error encoding batch: {e}")
        raise e


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen3-Embedding-8B") 
    parser.add_argument("--dataset-name", default="HTung/dialog_CC_clean_v2")
    parser.add_argument("--output-dir", default="./dataset") 
    parser.add_argument("--log-file", default="data.log")
    parser.add_argument("--output-file", default="train_dataset.parquet")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for encoding")
    parser.add_argument("--max-workers", type=int, default=16, help="Number of threads for parallel requests")
    parser.add_argument("--top-k", type=int, default=64) 
    parser.add_argument("--similarity-threshold", type=float, default=0.65)
    parser.add_argument("--max-seq-length", type=int, default=4096, help="Max tokens in history anchor")
    parser.add_argument("--triplet-batch-size", type=int, default=5000, help="Batch size for writing triplets")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(str(out_dir / args.log_file))
    output_path = out_dir / args.output_file

    embedding_model = Qwen3Embedding()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    logging.info(f"Loading Dataset: {args.dataset_name}")
    ds = load_dataset(args.dataset_name, split="train")
    
    logging.info(f"Processing dialogues (max_seq_length={args.max_seq_length})...")
    candidates = []
    unique_responses = {} 
    
    raw_data = ds["conversation_clean"]

    for idx, raw_dialog in enumerate(tqdm(raw_data, desc="Parsing")):
        try:
            dialog = read_dialog(raw_dialog)
            for i, turn in enumerate(dialog):
                if turn["role"] == "OP":
                    history = dialog[:i]
                    if not history: continue
                    
                    history_text = format_history(history, tokenizer)
                    
                    num_tokens = count_tokens(history_text, tokenizer)
                    if num_tokens > args.max_seq_length:
                        continue
                    
                    response_text = turn["text"]
                    
                    candidates.append({
                        "history": history_text,
                        "response_text": response_text,
                        "num_tokens": num_tokens
                    })
                    
                    if response_text not in unique_responses:
                        unique_responses[response_text] = None
        except Exception as e:
            continue

    logging.info(f"Found {len(candidates)} valid samples, {len(unique_responses)} unique responses")


    logging.info("Encoding Response Pool...")
    unique_response_list = list(unique_responses.keys())
    pool_embeddings = encode_in_batches(
        embedding_model, 
        unique_response_list, 
        batch_size=args.batch_size,
        max_workers=args.max_workers
    )
    
    text_to_idx = {text: i for i, text in enumerate(unique_response_list)}


    logging.info("Building FAISS Index...")
    index = faiss.IndexFlatIP(pool_embeddings.shape[1])
    index.add(pool_embeddings)


    logging.info("Encoding History (Anchor) Texts...")
    anchor_texts = [c["history"] for c in candidates]
    anchor_embeddings = encode_in_batches(
        embedding_model,
        anchor_texts,
        batch_size=args.batch_size,
        max_workers=args.max_workers
    )


    logging.info("Searching FAISS for Hard Negatives...")
    D, I = index.search(anchor_embeddings, k=args.top_k * 5)
    
    total_pool_size = len(unique_response_list)
    
    logging.info("Constructing Triplets with Batch Writing...")

    writer = None
    batches_written = 0
    current_batch = []

    logging.info(f"Writing triplets in batches of {args.triplet_batch_size}...")

    for i in tqdm(range(len(candidates)), desc="Finalizing"):
        item = candidates[i]
        pos_text = item["response_text"]
        
        if pos_text not in text_to_idx:
            continue
            
        pos_idx = text_to_idx[pos_text]
        pos_embedding = pool_embeddings[pos_idx]
        
        neighbor_indices = I[i]
        hard_neg_data = [] 
        
        for neg_idx in neighbor_indices:
            if neg_idx == pos_idx: 
                continue
            
            neg_embedding = pool_embeddings[neg_idx]
            sim = np.dot(pos_embedding, neg_embedding)
            
            if sim < args.similarity_threshold:
                neg_text = unique_response_list[neg_idx]
                hard_neg_data.append({
                    "text": neg_text,
                    "embedding": neg_embedding.tolist(),
                })
            
            if len(hard_neg_data) >= args.top_k:
                break
        
        # Fill remaining with random negatives if needed
        attempts = 0
        max_attempts = args.top_k * 10
        while len(hard_neg_data) < args.top_k and attempts < max_attempts:
            rand_idx = np.random.randint(0, total_pool_size)
            if rand_idx == pos_idx:
                attempts += 1
                continue
            
            rand_text = unique_response_list[rand_idx]
            rand_embedding = pool_embeddings[rand_idx]
            
            if rand_text not in [x["text"] for x in hard_neg_data]:
                hard_neg_data.append({
                    "text": rand_text,
                    "embedding": rand_embedding.tolist(),
                })
            
            attempts += 1

        hard_neg_data = hard_neg_data[:args.top_k]

        # Add to current batch
        current_batch.append({
            "history": item["history"],
            "positive_text": pos_text,
            "positive_embedding": pos_embedding.tolist(),
            "negative_texts": [x["text"] for x in hard_neg_data],
            "negative_embeddings": [x["embedding"] for x in hard_neg_data],
        })
        
        # Write batch when it reaches triplet_batch_size
        if len(current_batch) >= args.triplet_batch_size:
            df_batch = pd.DataFrame(current_batch)
            
            if writer is None:
                table = pa.Table.from_pandas(df_batch)
                writer = pq.ParquetWriter(output_path, table.schema)
            
            table = pa.Table.from_pandas(df_batch)
            writer.write_table(table)
            
            batches_written += 1
            logging.info(f"✓ Written batch {batches_written} ({len(current_batch)} triplets)")
            
            current_batch = []

    # Write remaining triplets
    if len(current_batch) > 0:
        df_batch = pd.DataFrame(current_batch)
        
        if writer is None:
            table = pa.Table.from_pandas(df_batch)
            writer = pq.ParquetWriter(output_path, table.schema)
        
        table = pa.Table.from_pandas(df_batch)
        writer.write_table(table)
        
        batches_written += 1
        logging.info(f"✓ Written final batch {batches_written} ({len(current_batch)} triplets)")

    if writer is not None:
        writer.close()

    logging.info(f"✓ Done! Total {batches_written} batches written to {output_path}")


if __name__ == "__main__":
    main()