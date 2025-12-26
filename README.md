# Annabelle

Train model LMM (Qwen/Qwen3-4B-Instruct-2507) + lớp cuối projection head 

## Nội dung chính

- `train.py` : script huấn luyện mô hình theo kiểu trainer cổ đại
- `train2.py` : script huấn luyện theo trainer có sẵn 
- `custom_llm_model.py` : (nếu có) định nghĩa mô hình tùy chỉnh.
- `qwen.py` : dựng sglang qwen để tạo data 
- `create_testdata.py` : tạo dữ liệu test thử nghiệm.
- `data.py` : tạo dữ liệu triplet (với anchor, pos, negs, pos_embedding, neg_embddings)
- `push_dataset.py`, `push_op_response_dataset.py` : các script để đẩy dataset HF.
- `test.py` : script kiểm thử / inference nhanh.
- `checkpoints/` : thư mục lưu checkpoint mô hình.


## Cài đặt nhanh

1. 

```bash
pip install -r requirments.txt
```

Nếu bạn dùng `pip` trong môi trường khác hoặc `conda`, điều chỉnh lệnh tương ứng.

## Cách chạy 
- tạo file .env giông .env.sample, hf key thì phải ở mode write 

- tạo data và đẩy dataset (Đã được tạo ở Ryuk00/annabelle):

```bash
python data.py 
python push_dataset.py
python push_op_response_dataset.py
```

- Tạo dữ liệu test (Đã được tạo ở Ryuk00/annabelle-response):

```bash
python create_testdata.py
```

- Huấn luyện (ví dụ):

```bash
python train.py
python train2.py
```

- Kiểm thử / inference nhanh:

```bash
python test.py
```
