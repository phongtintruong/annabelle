from huggingface_hub import HfApi
import os

# Cấu hình
FILE_PATH = "./sharded_data/data-0017.parquet" # Kiểm tra lại tên file trong thư mục của bạn
REPO_ID = "Ryuk00/annabelle"
HF_TOKEN = "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" # Token Write của bạn

def upload_last_file():
    api = HfApi(token=HF_TOKEN)
    
    if os.path.exists(FILE_PATH):
        print(f"Đang upload nốt file: {FILE_PATH}")
        try:
            api.upload_file(
                path_or_fileobj=FILE_PATH,
                path_in_repo="data/data-0017.parquet", # Đặt tên giống format các file trước
                repo_id=REPO_ID,
                repo_type="dataset"
            )
            print("Upload thành công file cuối!")
        except Exception as e:
            print(f"Vẫn lỗi: {e}")
            print("Hãy chắc chắn bạn đã chuyển Repo sang Public hoặc mua gói Pro.")
    else:
        print(f"Không tìm thấy file {FILE_PATH}. Kiểm tra lại thư mục.")

if __name__ == "__main__":
    upload_last_file()