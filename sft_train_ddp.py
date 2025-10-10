# for supervised fine-tuning
# from datasets import load_dataset
# from transformers import AutoTokenizer
#
#
# test_dataset = load_dataset("json", data_files="/home/jnu/jiananfu/project/GPT2/dataset/sft_mini_512.jsonl", split="train[:50]")
# print(test_dataset)
# print(test_dataset[0])
# tokenizer = AutoTokenizer.from_pretrained("/home/jnu/jiananfu/project/GPT2/HuggingFace/minimind_tokenizer")
#
#
# def format_prompt(text):
#     chat = test_dataset[0]['conversations']
#     prompt = tokenizer.apply_chat_template(chat, tokenize=False)
#     return {'text': prompt}
#
#
# dataset = test_dataset.map(format_prompt, remove_columns=test_dataset.column_names)
# print(dataset[0])
# for supervised fine-tuning
import os
import json
from typing import Dict
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

# ------------------ DDP 配置 (与您之前的代码一致) ------------------
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))
DEVICE = torch.device("cuda", LOCAL_RANK) if torch.cuda.is_available() else torch.device("cpu")
print(f"[RANK {RANK}] Using device {DEVICE}")

# ------------------ 1. 路径和常量 ------------------
# --- 修改为你自己的路径 ---
PRETRAINED_MODEL_PATH = "/home/jnu/jiananfu/project/GPT2/HuggingFace/model_save/checkpoint-60000"
TOKENIZER_PATH = "/home/jnu/jiananfu/project/GPT2/HuggingFace/minimind_tokenizer"
TRAIN_FILE_PATH = "/home/jnu/jiananfu/project/GPT2/dataset/sft_mini_512.jsonl"
OUTPUT_DIR = "sft_lora_model"
# -----------------------------

# ------------------ 2. 加载模型和 Tokenizer ------------------
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
model = GPT2LMHeadModel.from_pretrained(PRETRAINED_MODEL_PATH)

# ------------------ 3. LoRA 配置 (保持不变) ------------------
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.1,
    bias="none", target_modules=["c_attn"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ------------------ 4. 数据集处理 (利用Chat Template的逻辑重构) ------------------
class SFTDataset(Dataset):
    def __init__(self, file_path: str, tokenizer_d, max_length: int = 512):
        self.tokenizer = tokenizer_d
        self.max_length = max_length
        # 获取特殊token的ID，以便手动构建
        self.im_start_id = tokenizer_d.bos_token_id
        self.im_end_id = tokenizer_d.eos_token_id

        print(f"Loading data from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]
        print(f"Loaded {len(self.data)} examples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        conversations = item['conversations']

        input_ids = []
        labels = []

        # 使用 apply_chat_template 来获取每一轮的 token ids
        # 这样可以保证格式的绝对正确性
        for turn in conversations:
            # apply_chat_template 需要一个列表作为输入
            message = [turn]
            # tokenize=False 先获取格式化后的字符串
            # add_generation_prompt=False 确保我们得到的是历史记录格式
            formatted_text = self.tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=False
            )

            # 编码得到的字符串
            tokenized_output = self.tokenizer.encode(formatted_text, add_special_tokens=False)

            input_ids.extend(tokenized_output)

            # 根据角色创建 labels
            if turn['role'] == 'user':
                labels.extend([-100] * len(tokenized_output))
            else:  # assistant
                labels.extend(tokenized_output)

        # 截断
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ------------------ 5. 实例化数据集和 Data Collator ------------------
train_dataset = SFTDataset(
    file_path=TRAIN_FILE_PATH,
    tokenizer_d=tokenizer,
    max_length=512  # 根据您的GPU显存和文本长度调整
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=8
)

# ------------------ 6. 训练参数 (参考您的预训练配置) ------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    optim="adamw_torch",
    lr_scheduler_type='cosine',
    learning_rate=2e-4,                     # LoRA 微调可以使用稍大的学习率
    num_train_epochs=3,                     # SFT 通常 1-3 个 epoch 即可
    per_device_train_batch_size=8,          # SFT序列可能更长，batch size可能需要减小
    gradient_accumulation_steps=4,          # 等效 batch size = 8 * 4 * WORLD_SIZE
    weight_decay=0.01,
    warmup_ratio=0.03,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=3,
    logging_steps=100,
    fp16=True,
    report_to="tensorboard",
    logging_dir="sft_logs",
    dataloader_drop_last=True,
    # 如果内存不足，请减小此数值
    dataloader_num_workers=4,
    ddp_find_unused_parameters=False,
    run_name="GPT2_SFT_LoRA",
    save_safetensors=False,
)

# ------------------ 7. 初始化 Trainer 并开始训练 ------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("Starting SFT training with LoRA...")
trainer.train()

# ------------------ 8. 保存最终的LoRA适配器 ------------------
print("Saving final LoRA adapter...")
# PEFT 会自动处理，只保存 LoRA 的适配器权重
trainer.save_model(os.path.join(OUTPUT_DIR, "final_adapter"))
