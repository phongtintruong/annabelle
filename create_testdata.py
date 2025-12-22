import pandas as pd
import random
from datasets import load_dataset
import config

def extract_test_samples():
    print(f"Loading dataset from {config.DATASET_PATH}...")
    dataset = load_dataset(
        config.DATASET_PATH, 
        split="train", 
        token=config.HF_TOKEN, 
        verification_mode="no_checks"
    )
    
    total_rows = len(dataset)
    num_samples = 50
    
    if total_rows < num_samples:
        num_samples = total_rows
        
    print(f"Selecting {num_samples} random samples from {total_rows} rows...")
    random_indices = random.sample(range(total_rows), num_samples)
    subset = dataset.select(random_indices)
    
    cols = dataset.column_names
    history_col = "history" 
    response_col = "positive_text" if "positive_text" in cols else "response_text"
    
    data = []
    for item in subset:
        data.append({
            "anchor": item[history_col],
            "positive": item[response_col]
        })
        
    output_file = "test_data.csv"
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False, encoding="utf-8")
    
    print(f"Done. Saved to {output_file}")

if __name__ == "__main__":
    extract_test_samples()