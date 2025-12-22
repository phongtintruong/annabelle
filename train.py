import gc
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

import config

# --- CONFIG ---
MODEL_NAME = config.MODEL_NAME
DATASET_PATH = config.DATASET_PATH
TARGET_EMBEDDING_SIZE = config.TARGET_EMBEDDING_SIZE

BATCH_SIZE = config.BATCH_SIZE
LEARNING_RATE = config.LEARNING_RATE
WEIGHT_DECAY = config.WEIGHT_DECAY
NUM_EPOCHS = config.NUM_EPOCHS
MNRL_SCALE_FACTOR = config.MNRL_SCALE_FACTOR
WARMUP_RATIO = config.WARMUP_RATIO
MAX_SEQ_LENGTH = config.MAX_SEQ_LENGTH

OUTPUT_DIR = config.OUTPUT_DIR
SAVE_STEPS = config.SAVE_STEPS
WANDB_PROJECT = config.WANDB_PROJECT
HF_TOKEN = config.HF_TOKEN
WANDB_API_KEY = config.WANDB_API_KEY
HF_REPO_ID = config.HF_REPO_ID
PUSH_TO_HUB = config.PUSH_TO_HUB

GRADIENT_ACCUMULATION_STEPS = config.GRADIENT_ACCUMULATION_STEPS
MAX_GRAD_NORM = config.MAX_GRAD_NORM
USE_FLASH_ATTENTION = config.USE_FLASH_ATTENTION

# --- SETUP ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
        ).shuffle(seed=42)
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
        padding="max_length",
        truncation=False,
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
    
    model.qwen.save_pretrained(save_path)
    
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
model_wrapper.projection_head.float()
model_wrapper.qwen = prepare_model_for_kbit_training(
    model_wrapper.qwen,
    use_gradient_checkpointing=True
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION
)

model_wrapper.qwen = get_peft_model(model_wrapper.qwen, lora_config)

device = model_wrapper.qwen.device 
model_wrapper.projection_head.to(device) 


for param in model_wrapper.projection_head.parameters():
    param.requires_grad = True

print("\n" + "="*50)
model_wrapper.qwen.print_trainable_parameters()
print("="*50)


wandb.watch(model_wrapper, log="gradients", log_freq=100)
# --- TRAINING ---
dataset = ParquetDataset(DATASET_PATH)

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

num_batches_per_epoch = len(dataloader)
num_update_steps_per_epoch = (num_batches_per_epoch + GRADIENT_ACCUMULATION_STEPS - 1) // GRADIENT_ACCUMULATION_STEPS
total_optimization_steps = num_update_steps_per_epoch * NUM_EPOCHS
warmup_steps = int(total_optimization_steps * WARMUP_RATIO)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_optimization_steps,
    num_cycles=0.5,
    last_epoch=-1
)
loss_fn = HardNegativeInfoNCELoss(scale=MNRL_SCALE_FACTOR)

print("\n" + "="*50)
print("TRAINING CONFIG")
print("="*50)
print(f"Samples: {len(dataset)}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Grad accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"Batches per epoch: {num_batches_per_epoch}")
print(f"Update steps per epoch: {num_update_steps_per_epoch}")
print(f"Total optimization steps: {total_optimization_steps}")
print(f"Warmup steps: {warmup_steps}")
print(f"Device: {device}")
print("="*50 + "\n")

print("--- Starting Training ---\n")
model_wrapper.train()
global_step = 0

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    num_loss_accumulated = 0 
    accumulation_counter = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    
    for batch_idx, batch in enumerate(progress_bar):
        tokens = batch['tokens']
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)
        pos_embs = batch['positive_embeddings'].to(device)
        neg_embs = batch['negative_embeddings'].to(device)
        
        anchor_embs = model_wrapper(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(anchor_embs, pos_embs, neg_embs)
        
        loss = loss / GRADIENT_ACCUMULATION_STEPS
        loss.backward()
        
        accumulation_counter += 1
        
        current_loss_val = loss.item() * GRADIENT_ACCUMULATION_STEPS
        
        epoch_loss += current_loss_val
        num_loss_accumulated += 1
        
        is_accumulation_step = (accumulation_counter % GRADIENT_ACCUMULATION_STEPS == 0)
        is_last_batch = (batch_idx == len(dataloader) - 1)
        
        if is_accumulation_step or is_last_batch:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model_wrapper.parameters()),
                MAX_GRAD_NORM
            )
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            global_step += 1
            
            wandb.log({
                "train_loss": current_loss_val,
                "learning_rate": scheduler.get_last_lr()[0],
                "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                "epoch": epoch + 1,
                "step": global_step
            })
                        
            progress_bar.set_postfix({
                'loss': f'{current_loss_val:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                'gnorm': f'{grad_norm:.2f}' if isinstance(grad_norm, torch.Tensor) else f'{grad_norm:.2f}'
            })
            
            if global_step % SAVE_STEPS == 0:
                save_checkpoint(
                    model_wrapper, 
                    tokenizer, 
                    OUTPUT_DIR, 
                    f"checkpoint-{global_step}", 
                    push=PUSH_TO_HUB
                )
                torch.cuda.empty_cache()

        del input_ids, attention_mask, pos_embs, neg_embs, anchor_embs, loss
        
    avg_epoch_loss = epoch_loss / num_loss_accumulated
    
    print(f"\n{'='*50}")
    print(f"✓ Epoch {epoch + 1}/{NUM_EPOCHS} Complete")
    print(f"  Average Loss: {avg_epoch_loss:.4f}")
    print(f"  Batches Processed: {num_loss_accumulated}")
    print(f"  Optimization Steps: {global_step}")
    print(f"{'='*50}\n")
    
    save_checkpoint(
        model_wrapper, 
        tokenizer, 
        OUTPUT_DIR, 
        f"epoch-{epoch+1}", 
        push=PUSH_TO_HUB
    )
    
    gc.collect()
    torch.cuda.empty_cache()

print("\n" + "="*50)
print("✓ Training Finished!")
print(f"  Total Epochs: {NUM_EPOCHS}")
print(f"  Total Optimization Steps: {global_step}")
print("="*50)

save_checkpoint(
    model_wrapper, 
    tokenizer, 
    OUTPUT_DIR, 
    "final_model", 
    push=PUSH_TO_HUB
)

wandb.finish()