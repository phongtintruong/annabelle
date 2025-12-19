import os
from dotenv import load_dotenv

load_dotenv()

# --- CONFIG ---
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DATASET_PATH = "Ryuk00/annabelle" 
TARGET_EMBEDDING_SIZE = 4096

BATCH_SIZE = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 3
MNRL_SCALE_FACTOR = 20.0
WARMUP_RATIO = 0.05
MAX_SEQ_LENGTH = 4096

OUTPUT_DIR = "./checkpoints"
SAVE_STEPS = 200 
WANDB_PROJECT = "qwen-embedding-finetune"

HF_TOKEN = os.getenv("HF_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

HF_REPO_ID = "Ryuk00/qwen-llm-finetuned-v4"
PUSH_TO_HUB = True

GRADIENT_ACCUMULATION_STEPS = 16
MAX_GRAD_NORM = 1.0
USE_FLASH_ATTENTION = True

# Load Model Config
SUBFOLDER = "epoch-3"

# --- DATASET PUSH ---
DATASET_INPUT_FILE = "/workspace/bell_llm/dataset/train_dataset.parquet"
DATASET_REPO_ID = "Ryuk00/annabelle"
SHARDED_DATA_DIR = "./sharded_data"
CHUNK_SIZE_MB = 2000

# --- EMBEDDING SERVICE ---
EMBEDDING_BASE_URL = "http://localhost:30000/v1"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
EMBEDDING_TIMEOUT = 30.0
EMBEDDING_ENCODING_FORMAT = "float"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "abc")

# --- DATA PROCESSING ---
SOURCE_DATASET_NAME = "HTung/dialog_CC_clean_v2"
DATASET_OUTPUT_DIR = "./dataset"
DATA_LOG_FILE = "data.log"
DATA_OUTPUT_FILE = "train_dataset.parquet"
DATA_BATCH_SIZE = 32
DATA_MAX_WORKERS = 16
DATA_TOP_K = 64
DATA_SIMILARITY_THRESHOLD = 0.65
TRIPLET_BATCH_SIZE = 5000

# --- TEST ---
TEST_FILE_PATH = "./sharded_data/data-0017.parquet"
