import os
from torch.utils.data import IterableDataset, DataLoader
from transformers import (
    GPT2LMHeadModel, GPT2Config, AutoTokenizer, TrainingArguments, Trainer
)
from datasets import load_dataset
import torch

# ------------------ 分布式工具 ------------------
def get_local_rank():
    """获取 local_rank (DDP 下每个进程的 GPU id)"""
    return int(os.environ.get("LOCAL_RANK", 0))

def get_world_size():
    return int(os.environ.get("WORLD_SIZE", 1))

def get_rank():
    return int(os.environ.get("RANK", 0))

# ------------------ IterableDataset 支持 DDP ------------------
class HFIterableDatasetDDP(IterableDataset):
    """
    支持 DDP 的 IterableDataset，每个进程只处理自己的子序列
    """
    def __init__(self, dataset_stream, rank_d=0, world_size_d=1):
        self.dataset_stream = dataset_stream
        self.rank = rank_d
        self.world_size = world_size_d

    def __iter__(self):
        for idx, item in enumerate(self.dataset_stream):
            # DDP 按 idx 取模，保证不同 rank 取不同样本
            if idx % self.world_size == self.rank:
                yield item

# ------------------ Tokenization ------------------
def tokenize_function(example):
    enc = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_overflowing_tokens=True,
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # 展开溢出的块
    result = []
    for ids, mask in zip(input_ids, attention_mask):
        ids.append(tokenizer.eos_token_id)
        mask.append(1)
        ids = ids[:512]
        mask = mask[:512]
        result.append({
            "input_ids": ids,
            "attention_mask": mask,
            "labels": ids.copy()
        })
    return result

def tokenize_stream(dataset_s):
    for example in dataset_s:
        for enc in tokenize_function(example):
            yield enc

# ------------------ 初始化 tokenizer & model ------------------
tokenizer = AutoTokenizer.from_pretrained(
    r"C:\PytorchCode\GPT2\learningLLMs\minimind_tokenizer"
)
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=512,
    n_embd=768,
    n_layers=8,
    n_head=8,
)
model = GPT2LMHeadModel(config)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.bos_token_id = tokenizer.bos_token_id
# 修改生成配置
if hasattr(model, "generation_config"):
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id

# ------------------ 加载 streaming dataset ------------------
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
tokenized_dataset = tokenize_stream(dataset)

# 获取 DDP rank / world_size
local_rank = get_local_rank()
rank = get_rank()
world_size = get_world_size()

# 包装支持 DDP 的 IterableDataset
train_data = HFIterableDatasetDDP(tokenized_dataset, rank_d=rank, world_size_d=world_size)

# ------------------ TrainingArguments ------------------
training_args = TrainingArguments(
    output_dir="model_save",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=16,
    learning_rate=3e-4,
    num_train_epochs=1,
    logging_steps=100,
    save_strategy='steps',
    save_steps=500,
    save_total_limit=2,
    max_steps=20000,
    report_to="tensorboard",
    logging_dir="logs",
    disable_tqdm=False,
    fp16=True,
    ddp_find_unused_parameters=False,
)

# ------------------ Trainer ------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    tokenizer=tokenizer,
)

if __name__ == "__main__":
    trainer.train()
