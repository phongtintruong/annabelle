import pyarrow.parquet as pq
import pyarrow as pa
import os
from huggingface_hub import HfApi, create_repo

import config

# --- CẤU HÌNH ---
INPUT_FILE = config.DATASET_INPUT_FILE
REPO_ID = config.DATASET_REPO_ID
OUTPUT_DIR = config.SHARDED_DATA_DIR
CHUNK_SIZE_MB = config.CHUNK_SIZE_MB
HF_TOKEN = config.HF_TOKEN
# ----------------
def shard_upload_and_delete():
    api = HfApi(token=HF_TOKEN)
    
    try:
        create_repo(REPO_ID, repo_type="dataset", private=True, exist_ok=True, token=HF_TOKEN)
        print(f"Đã kết nối với repo: {REPO_ID}")
    except Exception as e:
        print(f"Lỗi kết nối repo (có thể đã tồn tại): {e}")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Đang mở file {INPUT_FILE}...")
    parquet_file = pq.ParquetFile(INPUT_FILE)
    schema = parquet_file.schema.to_arrow_schema()
    
    current_chunk_index = 0
    writer = None
    output_path = ""
    
    output_path = os.path.join(OUTPUT_DIR, f"data-{current_chunk_index:04d}.parquet")
    writer = pq.ParquetWriter(output_path, schema)
    
    print("Bắt đầu quy trình: Cắt -> Upload -> Xóa...")
    
    for i in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(i)
        writer.write_table(table)
        
        current_size = os.path.getsize(output_path)
        
        if current_size > CHUNK_SIZE_MB * 1024 * 1024:
            writer.close()
            print(f"-> Đã đóng gói: {os.path.basename(output_path)} ({current_size / (1024*1024):.2f} MB)")
            
            print(f"   Đang upload {os.path.basename(output_path)} lên Hugging Face...")
            try:
                api.upload_file(
                    path_or_fileobj=output_path,
                    path_in_repo=f"data/{os.path.basename(output_path)}", # Để trong thư mục data/
                    repo_id=REPO_ID,
                    repo_type="dataset"
                )
                print("   Upload thành công!")
                
                os.remove(output_path)
                print("   Đã xóa file tạm để giải phóng bộ nhớ.")
                
            except Exception as e:
                print(f"!!! Lỗi upload file {output_path}: {e}")
                return 

            current_chunk_index += 1
            output_path = os.path.join(OUTPUT_DIR, f"data-{current_chunk_index:04d}.parquet")
            writer = pq.ParquetWriter(output_path, schema)

    if writer:
        writer.close()
        if os.path.getsize(output_path) > 0:
            print(f"-> Đang xử lý file cuối: {os.path.basename(output_path)}")
            try:
                api.upload_file(
                    path_or_fileobj=output_path,
                    path_in_repo=f"data/{os.path.basename(output_path)}",
                    repo_id=REPO_ID,
                    repo_type="dataset"
                )
                print("   Upload file cuối thành công!")
                os.remove(output_path)
                print("   Đã xóa file cuối.")
            except Exception as e:
                print(f"!!! Lỗi upload file cuối: {e}")
        else:
            os.remove(output_path)

    print("=== HOÀN TẤT TOÀN BỘ QUÁ TRÌNH ===")

if __name__ == "__main__":
    if os.path.exists(OUTPUT_DIR):
        import shutil
        print("Đang dọn dẹp thư mục tạm cũ...")
        shutil.rmtree(OUTPUT_DIR)
        
    shard_upload_and_delete()