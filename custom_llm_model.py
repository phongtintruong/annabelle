import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class QwenForNextMessagePrediction(nn.Module):
    def __init__(self, model_name, output_embedding_size, **kwargs):
        super().__init__()
        self.qwen = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            trust_remote_code=True,
            **kwargs 
        )

        hidden_size = self.qwen.config.hidden_size 
  
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, output_embedding_size),
            nn.Tanh()
        )
        self.projection_head.float() 

    # def forward(self, input_ids, attention_mask):
    #     outputs = self.qwen.model(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask
    #     )
    #     last_hidden_state = outputs.last_hidden_state


    #     # Apply Last Token Pooling
    #     last_token_indices = attention_mask.sum(dim=1) - 1
    #     pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), last_token_indices]

    #     # Project to the final embedding space
    #     embedding = self.projection_head(pooled_output)
    #     return embedding

    def forward(self, input_ids, attention_mask):
        outputs = self.qwen(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        last_hidden_state = outputs.hidden_states[-1]
        last_token_indices = attention_mask.sum(dim=1) - 1
        
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), 
            last_token_indices
        ]

        pooled_output = pooled_output.to(torch.float32)
        
        embedding = self.projection_head(pooled_output)
        return embedding

# # --- Usage ---
# import config
# from huggingface_hub import HfApi, login

# MODEL_NAME = config.MODEL_NAME
# TARGET_EMBEDDING_SIZE = config.TARGET_EMBEDDING_SIZE
# HF_TOKEN = config.HF_TOKEN

# login(token=HF_TOKEN)
# model = QwenForNextMessagePrediction(
#     model_name=MODEL_NAME,
#     output_embedding_size=TARGET_EMBEDDING_SIZE
# )
# print("Custom model created successfully!")
# print("Projection head architecture:")
# print(model.projection_head)
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# prompt = "Give me a short introduction to large language model."
# messages = [
#     {"role": "user", "content": prompt}
# ]
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True,
# )
# model_inputs = tokenizer([text], return_tensors="pt")
# embed = model(**model_inputs)
# print(embed.shape)
# print(embed)