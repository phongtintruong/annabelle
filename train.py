import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import HfApi, login
import wandb

from custom_llm_model import QwenForNextMessagePrediction 

# --- CONFIG ---
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DATASET_PATH = "Ryuk00/annabelle" 
TARGET_EMBEDDING_SIZE = 4096

BATCH_SIZE = 1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 3
MNRL_SCALE_FACTOR = 20.0
WARMUP_RATIO = 0.05
MAX_SEQ_LENGTH = 4096

OUTPUT_DIR = "./checkpoints"
SAVE_STEPS = 1000
WANDB_PROJECT = "qwen-embedding-finetune"
HF_TOKEN = "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"
WANDB_API_KEY = "2f94fd9e02c96c007db857ed53b48fd5e6cd8428"
HF_REPO_ID = "Ryuk00/qwen-llm-finetuned-v4"
PUSH_TO_HUB = True

GRADIENT_ACCUMULATION_STEPS = 16
MAX_GRAD_NORM = 1.0
USE_FLASH_ATTENTION = True

# --- SETUP ---
wandb.login(key=WANDB_API_KEY)
login(token=HF_TOKEN)
wandb.init(project=WANDB_PROJECT, name=f"run-{MODEL_NAME.split('/')[-1]}")
api = HfApi()

if PUSH_TO_HUB:
    api.create_repo(repo_id=HF_REPO_ID, exist_ok=True, private=True)

# --- LOSS ---
class HardNegativeInfoNCELoss(nn.Module):
    def __init__(self, scale=20.0):
        super().__init__()
        self.scale = scale
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, anchor_embeddings, positive_embeddings, negative_embeddings):
        anchor = anchor_embeddings.to(torch.float32)
        pos = positive_embeddings.to(torch.float32)
        neg = negative_embeddings.to(torch.float32)

        anchor = F.normalize(anchor, p=2, dim=1)
        pos = F.normalize(pos, p=2, dim=1)
        neg = F.normalize(neg, p=2, dim=2)

        pos_sim = (anchor * pos).sum(dim=1, keepdim=True)                    # [B, 1]
        neg_sim = torch.bmm(neg, anchor.unsqueeze(2)).squeeze(2)             # [B, N]

        logits = torch.cat([pos_sim, neg_sim], dim=1) * self.scale
        labels = torch.zeros(anchor.size(0), dtype=torch.long, device=logits.device)
        return self.cross_entropy(logits, labels)

# --- DATASET ---
class ParquetDataset(Dataset):
    def __init__(self, dataset_repo_id):
        print(f"Loading data from Hugging Face Hub: {dataset_repo_id}...")
        self.dataset = load_dataset(
            dataset_repo_id, 
            data_files="data/*.parquet", 
            split="train",
            token=HF_TOKEN,
            verification_mode="no_checks",          
        )
        print(f"Loaded {len(self.dataset)} samples from HF Hub")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "history": item['history'],
            "positive_embedding": torch.tensor(item['positive_embedding'], dtype=torch.float32),
            "negative_embeddings": torch.tensor(item['negative_embeddings'], dtype=torch.float32)
        }


def collate_fn(batch, tokenizer):
    histories = [item['history'] for item in batch]
    pos_embs = torch.stack([item['positive_embedding'] for item in batch])
    neg_embs = torch.stack([item['negative_embeddings'] for item in batch])
    
    tokens = tokenizer(
        histories,
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt"
    )
    return {
        "tokens": tokens,
        "positive_embeddings": pos_embs,
        "negative_embeddings": neg_embs
    }

def save_checkpoint(model, tokenizer, output_dir, step_name, push=False):
    save_path = os.path.join(output_dir, step_name)
    os.makedirs(save_path, exist_ok=True)
    
    model.backbone.save_pretrained(save_path)
    torch.save(model.projection_head.state_dict(), os.path.join(save_path, "head.pt"))
    tokenizer.save_pretrained(save_path)
    
    print(f"✓ Saved checkpoint to {save_path}")

    if push:
        print(f"Pushing {step_name} to HF Hub...")
        try:
            api.upload_folder(folder_path=save_path, repo_id=HF_REPO_ID, path_in_repo=step_name)
            print("✓ Push complete.")
        except Exception as e:
            print(f"✗ Failed: {e}")

# --- MODEL SETUP ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loading model...")
model_wrapper = QwenForNextMessagePrediction(
    model_name=MODEL_NAME,
    output_embedding_size=TARGET_EMBEDDING_SIZE,
    quantization_config=bnb_config,
    attn_implementation="flash_attention_2" if USE_FLASH_ATTENTION else "eager"
)

model_wrapper.backbone = prepare_model_for_kbit_training(
    model_wrapper.backbone,
    use_gradient_checkpointing=True
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION
)

model_wrapper.backbone = get_peft_model(model_wrapper.backbone, lora_config)
device = model_wrapper.backbone.device 
model_wrapper.projection_head.to(device) 


# ✅ PROJECTION HEAD TRAINABLE
for param in model_wrapper.projection_head.parameters():
    param.requires_grad = True

print("\n" + "="*50)
model_wrapper.backbone.print_trainable_parameters()
print("="*50)

# --- TRAINING ---
dataset = ParquetDataset(DATASET_PATH)

# preview the dataset
print(dataset[0])

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda b: collate_fn(b, tokenizer),
    num_workers=4,
    pin_memory=True
)

optimizer = AdamW(
    filter(lambda p: p.requires_grad, model_wrapper.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

total_steps = len(dataloader) * NUM_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
    num_cycles=0.5,
    last_epoch=-1
)

loss_fn = HardNegativeInfoNCELoss(scale=MNRL_SCALE_FACTOR)

print("\n" + "="*50)
print("TRAINING CONFIG")
print("="*50)
print(f"Samples: {len(dataset)}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Grad accumulation: {GRADIENT_ACCUMULATION_STEPS}")
print(f"Effective batch: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"Total steps: {total_steps}")
print(f"Warmup: {warmup_steps} steps")
print("="*50 + "\n")

print("--- Starting Training ---\n")
model_wrapper.train()
global_step = 0
accumulation_counter = 0

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    
    for batch_idx, batch in enumerate(progress_bar):
        tokens = batch['tokens']
        input_ids = tokens['input_ids'].to(model_wrapper.backbone.device)
        attention_mask = tokens['attention_mask'].to(model_wrapper.backbone.device)
        
        pos_embs = batch['positive_embeddings'].to(model_wrapper.backbone.device)
        neg_embs = batch['negative_embeddings'].to(model_wrapper.backbone.device)
        
        anchor_embs = model_wrapper(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(anchor_embs, pos_embs, neg_embs)
        
        loss = loss / GRADIENT_ACCUMULATION_STEPS
        loss.backward()
        accumulation_counter += 1
        
        if accumulation_counter % GRADIENT_ACCUMULATION_STEPS == 0 or (batch_idx == len(dataloader) - 1):
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model_wrapper.parameters()),
                MAX_GRAD_NORM
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            
            current_loss = loss.item() * GRADIENT_ACCUMULATION_STEPS
            epoch_loss += current_loss
            
            wandb.log({
                "train_loss": current_loss,
                "learning_rate": scheduler.get_last_lr()[0],
                "step": global_step
            })
            progress_bar.set_postfix({'loss': f'{current_loss:.4f}'})
            
            if global_step % SAVE_STEPS == 0:
                save_checkpoint(model_wrapper, tokenizer, OUTPUT_DIR, f"checkpoint-{global_step}", push=PUSH_TO_HUB)

    avg_loss = epoch_loss / (len(dataloader) // GRADIENT_ACCUMULATION_STEPS)
    print(f"\n✓ Epoch {epoch + 1} Avg Loss: {avg_loss:.4f}\n")
    save_checkpoint(model_wrapper, tokenizer, OUTPUT_DIR, f"epoch-{epoch+1}", push=PUSH_TO_HUB)

print("\n" + "="*50)
print("✓ Training Finished!")
print("="*50)
wandb.finish()