# train_gpt2_ddp_eos.py

import os
import torch
from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import IterableDataset
import json

# ------------------ DDP 配置 ------------------
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))
DEVICE = torch.device("cuda", LOCAL_RANK) if torch.cuda.is_available() else torch.device("cpu")
print(f"[RANK {RANK}] Using device {DEVICE}")


# ------------------ Dataset ------------------
class JsonlIterableDataset(IterableDataset):
    """
    支持：
    - 长文本分块（block_size）
    - attention_mask
    - 每块末尾加 eos_token，总长度保持 block_size
    - DDP + 多worker切片
    """
    def __init__(self, file_path, tokenizer, block_size=512, rank=0, world_size=1):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.rank = rank
        self.world_size = world_size
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        total_workers = worker_info.num_workers if worker_info else 1

        with open(self.file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx % (self.world_size * total_workers) != (self.rank * total_workers + worker_id):
                    continue
                data = json.loads(line.strip())
                input_ids_full = self.tokenizer.encode(data['text'])

                for i in range(0, len(input_ids_full), self.block_size):
                    chunk = input_ids_full[i:i+self.block_size]

                    # 在块末尾加 eos_token
                    if len(chunk) >= self.block_size:
                        # 块满了，替换最后一个 token 为 eos_token
                        chunk[-1] = self.pad_token_id
                        attention_mask = [1] * self.block_size
                    else:
                        # 添加 eos_token
                        chunk.append(self.pad_token_id)
                        attention_mask = [1] * len(chunk)
                        # 如果长度超过 block_size，需要 truncate
                        if len(chunk) > self.block_size:
                            chunk = chunk[:self.block_size]
                            attention_mask = [1] * self.block_size
                        # pad 到 block_size
                        if len(chunk) < self.block_size:
                            pad_len = self.block_size - len(chunk)
                            chunk += [self.tokenizer.pad_token_id] * pad_len
                            attention_mask += [0] * pad_len

                    yield {
                        'input_ids': chunk,
                        'labels': chunk.copy(),
                        'attention_mask': attention_mask
                    }


# ------------------ Tokenizer & Model ------------------
tokenizer = AutoTokenizer.from_pretrained("/home/jnu/jiananfu/project/GPT2/HuggingFace/minimind_tokenizer")

model_config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=512,
    n_embd=768,
    n_layer=12,
    n_head=12
)
model = GPT2LMHeadModel(model_config).to(DEVICE)

# ------------------ Dataset 实例 ------------------
dataset = JsonlIterableDataset(
    file_path="/home/jnu/jiananfu/project/GPT2/dataset/mobvoi_seq_monkey_general_open_corpus.jsonl",
    tokenizer=tokenizer,
    block_size=512,
    rank=RANK,
    world_size=WORLD_SIZE
)

# for i, batch in enumerate(dataset):
#     # print(batch['input_ids'])
#     print(len(batch['input_ids']))
#     # print(tokenizer.decode(batch['input_ids']))
#     if i == 99:
#         print(tokenizer.decode(batch['input_ids']))
#         break


# ------------------ Data Collator ------------------
def data_collator(batch):
    return {
        'input_ids': torch.tensor([f['input_ids'] for f in batch], dtype=torch.long),
        'labels': torch.tensor([f['labels'] for f in batch], dtype=torch.long),
        'attention_mask': torch.tensor([f['attention_mask'] for f in batch], dtype=torch.long)
    }


# ------------------ TrainingArguments ------------------
training_args = TrainingArguments(
    output_dir="model_save",
    overwrite_output_dir=True,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=2,
    fp16=True,
    num_train_epochs=1,
    save_strategy="steps",
    max_steps=60_000,
    save_steps=1000,
    save_total_limit=4,
    logging_steps=200,
    report_to="tensorboard",
    logging_dir="logs",
    dataloader_drop_last=True,
    dataloader_num_workers=8,
    ddp_find_unused_parameters=False,
    run_name="GPT2_pretrain"
)

# ------------------ Trainer ------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    data_collator=data_collator
)

# ------------------ 开始训练 ------------------
trainer.train()
