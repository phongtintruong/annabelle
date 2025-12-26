import logging
import random
import os
import json
import torch
from pathlib import Path
from torch import nn
from datasets import load_dataset
from huggingface_hub import login as hf_login
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
import wandb
from dotenv import load_dotenv
from transformers import BitsAndBytesConfig

from sentence_transformers import (
    SentenceTransformer, 
    SentenceTransformerTrainer, 
    SentenceTransformerTrainingArguments,
    losses,
    models,
    util 
)

import config

load_dotenv()

import torch
from torch import nn
import torch.nn.functional as F

import torch
from torch import nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util

class IndependentNegativesRankingLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, scale: float = 20.0):
        super(IndependentNegativesRankingLoss, self).__init__()
        self.model = model
        self.scale = scale
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, sentence_features, labels=None):
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]

        anchors = embeddings[0]  # [Batch, Hidden]
        
        candidates = torch.stack(embeddings[1:], dim=1)

        anchors = F.normalize(anchors, p=2, dim=1)       # [Batch, Hidden]
        candidates = F.normalize(candidates, p=2, dim=2) # [Batch, 1+K, Hidden]

        scores = torch.bmm(
            anchors.unsqueeze(1), 
            candidates.transpose(1, 2)
        ).squeeze(1)

        scores = scores * self.scale

        target_labels = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
        
        return self.cross_entropy_loss(scores, target_labels)
        
    def get_config_dict(self):
        return {"scale": self.scale}

class QwenLastTokenPooling(nn.Module):
    def __init__(self, word_embedding_dimension: int):
        super(QwenLastTokenPooling, self).__init__()
        self.word_embedding_dimension = word_embedding_dimension

    def forward(self, features):
        token_embeddings = features['token_embeddings']
        attention_mask = features['attention_mask']
        last_token_indices = attention_mask.sum(dim=1) - 1
        batch_size = token_embeddings.shape[0]
        pooled_output = token_embeddings[
            torch.arange(batch_size, device=token_embeddings.device),
            last_token_indices
        ]
        features.update({'sentence_embedding': pooled_output})
        return features

    def get_sentence_embedding_dimension(self):
        return self.word_embedding_dimension

    def get_config_dict(self):
        return {'word_embedding_dimension': self.word_embedding_dimension}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)
        return QwenLastTokenPooling(**config)

class Float32Dense(models.Dense):
    """Class Dense bắt buộc chạy trên Float32 để tránh lỗi loss NaN hoặc giảm độ chính xác khi train bf16"""
    def __init__(self, in_features, out_features, bias=True, activation_function=nn.Identity()):
        super().__init__(in_features, out_features, bias, activation_function)
        self.linear.to(torch.float32)

    def forward(self, features):
        input_vectors = features['sentence_embedding']
        input_vectors = input_vectors.to(torch.float32)
        if self.linear.weight.dtype != torch.float32:
            self.linear.to(torch.float32)
        
        with torch.autocast(device_type=input_vectors.device.type, enabled=False):
            output_vectors = self.linear(input_vectors)
            output_vectors = self.activation_function(output_vectors)
            
        features.update({'sentence_embedding': output_vectors})
        return features

    @staticmethod
    def load(input_path):
        dense = models.Dense.load(input_path)
        return Float32Dense(
            in_features=dense.linear.in_features,
            out_features=dense.linear.out_features,
            bias=dense.linear.bias is not None,
            activation_function=dense.activation_function
        )

class QwenSBERT(SentenceTransformer):
    def __init__(self, model_name_or_path, output_embedding_size=768, **kwargs):
        word_embedding_model = models.Transformer(
            model_name_or_path,
            max_seq_length=kwargs.get("max_seq_length", 1024),
            model_args={
                "trust_remote_code": True, 
                "dtype": "auto", 
                **kwargs.get("model_args", {})
            }
        )
        hidden_size = word_embedding_model.get_word_embedding_dimension()
        pooling_model = QwenLastTokenPooling(word_embedding_dimension=hidden_size)
        dense_model = Float32Dense(
            in_features=hidden_size,
            out_features=output_embedding_size,
            bias=True,
            activation_function=nn.Tanh()
        )
        modules = [word_embedding_model, pooling_model, dense_model]
        super().__init__(modules=modules)

MAX_NEGS = 64 

def prepare_dataset(dataset_repo_id):
    print(f"Loading data from Hugging Face Hub: {dataset_repo_id}...")
    
    raw_dataset = load_dataset(
        dataset_repo_id, 
        data_files="**/*.parquet", 
        split="train", 
        token=config.HF_TOKEN,
        verification_mode="no_checks"
    )

    def expand_samples(batch):
        outputs = {
            "anchor": [],
            "positive": []
        }
        for i in range(MAX_NEGS):
            outputs[f"negative_{i}"] = []

        for history, pos_text, all_negs in zip(batch['history'], batch['positive_text'], batch['negative_texts']):
            selected_negs = all_negs[:MAX_NEGS]
            
            if len(selected_negs) < MAX_NEGS:
                needed = MAX_NEGS - len(selected_negs)
                if len(all_negs) > 0:
                    selected_negs.extend(random.choices(all_negs, k=needed))
                else:
                    selected_negs.extend([""] * needed)

            outputs["anchor"].append(history)
            outputs["positive"].append(pos_text)
            
            for i, neg in enumerate(selected_negs):
                outputs[f"negative_{i}"].append(neg)
        
        return outputs

    print("Processing dataset: Mapping all negatives to columns...")
    processed_dataset = raw_dataset.map(
        expand_samples,
        batched=True,
        remove_columns=raw_dataset.column_names, 
        desc="Formatting dataset"
    )
    
    print(f"Original samples: {len(raw_dataset)}")
    print(f"Processed samples: {len(processed_dataset)}") 
    
    return processed_dataset


def main():
    out_dir = Path(config.OUTPUT_DIR)
    if config.WANDB_API_KEY:
        wandb.login(key=config.WANDB_API_KEY)
        wandb.init(project=config.WANDB_PROJECT, name="qwen-sbert-trainer")
    if config.PUSH_TO_HUB:
        hf_login(token=config.HF_TOKEN)
        
    random.seed(42)
    torch.manual_seed(42)

    train_dataset = prepare_dataset(config.DATASET_PATH)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    logging.info(f"Loading SBERT model: {config.MODEL_NAME}")
    
    model = QwenSBERT(
        model_name_or_path=config.MODEL_NAME,
        output_embedding_size=config.TARGET_EMBEDDING_SIZE,
        max_seq_length=config.MAX_SEQ_LENGTH,
        model_args={
            "quantization_config": bnb_config,
            "attn_implementation": "flash_attention_2" if config.USE_FLASH_ATTENTION else "eager"
        }
    )

    transformer_module = model[0] 
    transformer_module.auto_model = prepare_model_for_kbit_training(
        transformer_module.auto_model, 
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
    
    transformer_module.auto_model = get_peft_model(transformer_module.auto_model, lora_config)
    


    train_loss = IndependentNegativesRankingLoss(
        model=model,
        scale=config.MNRL_SCALE_FACTOR, 
    )

    args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        warmup_ratio=config.WARMUP_RATIO,
        fp16=False,
        bf16=True, 
        logging_steps=10,
        report_to="wandb" if config.WANDB_API_KEY else "none",
        run_name="qwen-sbert-run",
        save_strategy="steps",
        save_steps=config.SAVE_STEPS,
        max_grad_norm=config.MAX_GRAD_NORM,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        push_to_hub=config.PUSH_TO_HUB,
        hub_model_id=config.HF_REPO_ID,
        hub_token=config.HF_TOKEN,
        hub_strategy="every_save",
        remove_unused_columns=False, 
    )

    # F. Trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=train_loss,
    )

    logging.info("Starting training...")
    trainer.train()

    # Save
    logging.info("Saving model...")
    model.save_pretrained(str(out_dir / "final_model"))
    if hasattr(model[0], "tokenizer"):
         model[0].tokenizer.save_pretrained(str(out_dir / "final_model"))

    if config.PUSH_TO_HUB:
        trainer.push_to_hub(commit_message="Training completed")

    if config.WANDB_API_KEY:
        wandb.finish()

if __name__ == "__main__":
    main()