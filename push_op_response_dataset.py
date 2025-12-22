import os
import shutil
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfApi, create_repo
from tqdm import tqdm
import config

INPUT_DATASET_ID = config.DATASET_PATH
TARGET_REPO_ID = config.CORPUS_REPO_ID
TEMP_OUTPUT_DIR = config.SHARDED_DATA_DIR
CHUNK_SIZE_MB = config.CHUNK_SIZE_MB
HF_TOKEN = config.HF_TOKEN

def main():
    api = HfApi(token=HF_TOKEN)
    
    try:
        create_repo(TARGET_REPO_ID, repo_type="dataset", private=True, exist_ok=True, token=HF_TOKEN)
    except Exception as e:
        print(f"Repo check: {e}")

    if os.path.exists(TEMP_OUTPUT_DIR):
        shutil.rmtree(TEMP_OUTPUT_DIR)
    os.makedirs(TEMP_OUTPUT_DIR)

    dataset = load_dataset(INPUT_DATASET_ID, split="train", token=HF_TOKEN, verification_mode="no_checks")
    
    total_rows = len(dataset)
    batch_size = 1000
    
    seen_responses = set()
    
    current_buffer = []
    current_file_index = 0
    current_buffer_size_est = 0
    target_chunk_size_bytes = CHUNK_SIZE_MB * 1024 * 1024

    cols = dataset.column_names
    text_col = "positive_text" if "positive_text" in cols else "response_text"
    emb_col = "positive_embedding"

    for i in tqdm(range(0, total_rows, batch_size), desc="Processing"):
        batch = dataset[i : i + batch_size]
        
        texts = batch[text_col]
        embs = batch[emb_col]
        
        for text, emb in zip(texts, embs):
            if text in seen_responses:
                continue
            
            seen_responses.add(text)
            
            row = {
                "response": text,
                "embedding": emb
            }
            current_buffer.append(row)
            
            current_buffer_size_est += len(text.encode('utf-8')) + (len(emb) * 4)

            if current_buffer_size_est >= target_chunk_size_bytes:
                save_and_upload(api, current_buffer, current_file_index)
                current_buffer = []
                current_buffer_size_est = 0
                current_file_index += 1

    if current_buffer:
        save_and_upload(api, current_buffer, current_file_index)

    if os.path.exists(TEMP_OUTPUT_DIR):
        shutil.rmtree(TEMP_OUTPUT_DIR)

def save_and_upload(api, data, index):
    df = pd.DataFrame(data)
    table = pa.Table.from_pandas(df)
    
    filename = f"corpus-{index:04d}.parquet"
    filepath = os.path.join(TEMP_OUTPUT_DIR, filename)
    
    pq.write_table(table, filepath)
    
    api.upload_file(
        path_or_fileobj=filepath,
        path_in_repo=f"data/{filename}",
        repo_id=TARGET_REPO_ID,
        repo_type="dataset"
    )
    
    os.remove(filepath)

if __name__ == "__main__":
    main()