import os
from torch.utils.data import IterableDataset, DataLoader
from transformers import (
    GPT2LMHeadModel, GPT2Config, AutoTokenizer, TrainingArguments, Trainer
)
from datasets import load_dataset
import torch

# ------------------ DDP 配置 ------------------
def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))

def get_world_size():
    return int(os.environ.get("WORLD_SIZE", 1))

def get_rank():
    return int(os.environ.get("RANK", 0))

# ------------------ IterableDataset + DDP ------------------
class HFIterableDatasetDDP(IterableDataset):
    """
    支持 DDP 的 IterableDataset，每个进程只处理自己的子序列
    """
    def __init__(self, dataset_stream, rank=0, world_size=1):
        self.dataset_stream = dataset_stream
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        for idx, item in enumerate(self.dataset_stream):
            if idx % self.world_size == self.rank:
                yield item

# ------------------ Tokenization ------------------
def tokenize_batch(texts, tokenizer, max_length=512):
    enc = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    enc["labels"] = enc["input_ids"].clone()
    return enc

# ------------------ Collate 函数 ------------------
def collate_fn(batch):
    texts = [x["text"] for x in batch]
    return tokenize_batch(texts, tokenizer, max_length=512)

# ------------------ 初始化 tokenizer & model ------------------
tokenizer = AutoTokenizer.from_pretrained(
    r"C:\PytorchCode\GPT2\learningLLMs\minimind_tokenizer"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=512,
    n_embd=768,
    n_layers=8,
    n_head=8,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)
model = GPT2LMHeadModel(config)
# 同步 generation_config
if hasattr(model, "generation_config"):
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

# ------------------ 加载 JSONL 流式 dataset ------------------
data_files = {
    "train": [
        r"C:\PytorchCode\GPT2\dataset\split_jsonl\part_0.jsonl",
        r"C:\PytorchCode\GPT2\dataset\split_jsonl\part_1.jsonl",
    ]
}
dataset = load_dataset(
    'json',
    data_files=data_files,
    streaming=True,
    split="train",
)

# 获取 DDP rank / world_size
local_rank = get_local_rank()
rank = get_rank()
world_size = get_world_size()

train_data = HFIterableDatasetDDP(dataset, rank=rank, world_size=world_size)

# ------------------ TrainingArguments ------------------
training_args = TrainingArguments(
    output_dir="model_save",
    per_device_train_batch_size=8,  # 每张卡 batch_size，可以调节
    gradient_accumulation_steps=32, # 累积步数，总全局 batch = 8*2*32=512
    learning_rate=1e-4,
    num_train_epochs=2,
    logging_steps=50,
    save_strategy='steps',
    save_steps=500,
    save_total_limit=2,
    max_steps=20000,
    report_to="tensorboard",
    logging_dir="logs",
    disable_tqdm=False,
    fp16=True,                       # 混合精度
    ddp_find_unused_parameters=False,
)

# ------------------ DataLoader ------------------
train_dataloader = DataLoader(
    train_data,
    batch_size=training_args.per_device_train_batch_size,
    num_workers=18,          # CPU 核心数，可调
    pin_memory=True,
    collate_fn=collate_fn,
    prefetch_factor=4
)

# ------------------ Trainer ------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    tokenizer=tokenizer,
    data_collator=collate_fn
)

if __name__ == "__main__":
    trainer.train()
