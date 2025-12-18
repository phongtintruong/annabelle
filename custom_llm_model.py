import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class QwenForNextMessagePrediction(nn.Module):
    def __init__(self, model_name, output_embedding_size=4096, **kwargs):
        super().__init__()
        
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            trust_remote_code=True,
            **kwargs 
        )

        print(self.backbone)
        
        hidden_size = self.backbone.config.hidden_size 

        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, output_embedding_size),
            nn.Tanh()
        )

        # if hasattr(self.backbone, "dtype"):
        #     self.projection_head.to(dtype=self.backbone.dtype)

    def forward(self, input_ids, attention_mask):
        device = self.backbone.device

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        outputs = self.backbone.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        print(outputs)
        print(outputs.keys())
        last_hidden_state = outputs.last_hidden_state

        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(last_hidden_state.shape[0], device=device)
        pooled_output = last_hidden_state[batch_indices, sequence_lengths]

        embedding = self.projection_head(pooled_output.to(torch.float32))

        return embedding



# MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct" 
# TARGET_EMBEDDING_SIZE = 4096

# print(f"Loading model: {MODEL_NAME}...")
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = QwenForNextMessagePrediction(MODEL_NAME, output_embedding_size=TARGET_EMBEDDING_SIZE)

# if torch.cuda.is_available():
#     model.cuda()

# model.eval()

# messages = [
#     {"role": "system", "content": "You are a helpful AI."},
#     {"role": "user", "content": "Hello, generate an embedding for this sentence."}
# ]

# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True
# )

# print(f"\nInput Text:\n{text}")

# inputs = tokenizer(text, return_tensors="pt")

# with torch.no_grad():
#     embedding = model(inputs.input_ids, inputs.attention_mask)

# print("-" * 30)
# print(f"Output Shape: {embedding.shape}")


# if embedding.shape[1] == TARGET_EMBEDDING_SIZE:
#     print("\nSUCCESS: Output dimension matches target.")
# else:
#     print("\nFAIL: Dimension mismatch.")