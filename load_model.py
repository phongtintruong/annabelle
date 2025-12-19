import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import hf_hub_download
from custom_llm_model import QwenForNextMessagePrediction 

import config

# --- CẤU HÌNH ---
HF_REPO_ID = config.HF_REPO_ID 
SUBFOLDER = config.SUBFOLDER                           
MODEL_NAME = config.MODEL_NAME         
HF_TOKEN = config.HF_TOKEN 

def load_model_from_hub():
    print(f"Loading from {HF_REPO_ID} (Folder: {SUBFOLDER})...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("Initializing Base Model...")
    model = QwenForNextMessagePrediction(
        model_name=MODEL_NAME,
        output_embedding_size=4096,
        # quantization_config=bnb_config 
    )

    # 3. Load LoRA Adapter từ Hugging Face
    # Peft tự động tìm trong repo/subfolder để tải adapter_config.json và adapter_model.safetensors
    print("Loading LoRA Adapters...")
    model.backbone = PeftModel.from_pretrained(
        model.backbone,
        model_id=HF_REPO_ID,
        subfolder=SUBFOLDER,
        token=HF_TOKEN
    )

    # 4. Download và Load Custom Head (head.pt)
    print("Downloading & Loading Projection Head...")
    head_file_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="head.pt",
        subfolder=SUBFOLDER,
        token=HF_TOKEN
    )
    
    # Load weight vào lớp projection_head
    state_dict = torch.load(head_file_path, map_location="cpu")
    model.projection_head.load_state_dict(state_dict)

    # 5. Load Tokenizer (Cũng nằm trong cùng folder trên HF)
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        HF_REPO_ID,
        subfolder=SUBFOLDER,
        token=HF_TOKEN
    )

    # Đưa model vào chế độ Eval và chuyển sang GPU
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    
    return model, tokenizer

# --- SỬ DỤNG ---
if __name__ == "__main__":
    model, tokenizer = load_model_from_hub()
    
    # Test thử
    prompt = "Hello, nice to meet you"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(text, return_tensors="pt").to(model.backbone.device)
    
    with torch.no_grad():
        embedding = model(inputs.input_ids, inputs.attention_mask)
        
    print(f"\nSuccess! Embedding shape: {embedding.shape}")
    print(embedding[0, :10]) # In thử 10 số đầu