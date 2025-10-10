# for supervised fine-tuning
import os
import json
from dataclasses import dataclass
from typing import Dict, List

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
# ... (其余DDP配置保持不变) ...
DEVICE = torch.device("cuda", LOCAL_RANK) if torch.cuda.is_available() else torch.device("cpu")

# ------------------ 1. 路径和常量 ------------------
# --- 修改为你自己的路径 ---
PRETRAINED_MODEL_PATH = "/home/jnu/jiananfu/project/GPT2/HuggingFace/model_save/checkpoint-60000"
TOKENIZER_PATH = "/home/jnu/jiananfu/project/GPT2/HuggingFace/minimind_tokenizer"
TRAIN_FILE_PATH = "/path/to/your/1.5GB_sft_dataset.jsonl"
OUTPUT_DIR = "sft_lora_model_output"
# -----------------------------

# ------------------ 2. 加载模型和 Tokenizer ------------------
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
model = GPT2LMHeadModel.from_pretrained(PRETRAINED_MODEL_PATH)

# --- 验证特殊 token ---
# 您的 tokenizer 已有所有需要的 token，这里无需额外设置
# vocab_ids: <|im_start|>: 1, <|im_end|>: 2, <|endoftext|>: 0
# 确保 tokenizer 的 pad_token 也被正确设置，如果没有，则设置一个
if tokenizer.pad_token is None:
    # 尽管 <|endoftext|> 的 id 是 0，但通常我们用它来做 pad token
    tokenizer.pad_token = tokenizer.eos_token # eos_token 通常就是 <|endoftext|>
    model.config.pad_token_id = tokenizer.eos_token_id

# ------------------ 3. LoRA 配置 (保持不变) ------------------
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.1,
    bias="none", target_modules=["c_attn"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ------------------ 4. 数据集处理 (利用Chat Template的逻辑重构) ------------------
class SFTDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: AutoTokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 获取特殊token的ID，以便手动构建
        self.im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        
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
            else: # assistant
                labels.extend(tokenized_output)
        
        # 截断
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

# ------------------ 5. 实例化数据集和 Data Collator (保持不变) ------------------
train_dataset = SFTDataset(
    file_path=TRAIN_FILE_PATH,
    tokenizer=tokenizer,
    max_length=1024
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, model=model,
    label_pad_token_id=-100, pad_to_multiple_of=8
)

# ------------------ 6. 训练参数 (保持不变) ------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    # ... (其余参数与上一版代码完全相同) ...
    num_train_epochs=3, per_device_train_batch_size=8, gradient_accumulation_steps=4,
    learning_rate=2e-4, weight_decay=0.01, warmup_ratio=0.03, lr_scheduler_type="cosine",
    save_strategy="steps", save_steps=500, save_total_limit=3, logging_steps=50,
    fp16=True, report_to="tensorboard", logging_dir="sft_logs",
    dataloader_drop_last=True, dataloader_num_workers=8,
    ddp_find_unused_parameters=False, run_name="GPT2_SFT_LoRA_ChatTemplate",
    save_safetensors=False,
)

# ------------------ 7. 初始化 Trainer 并开始训练 (保持不变) ------------------
trainer = Trainer(
    model=model, args=training_args, train_dataset=train_dataset,
    tokenizer=tokenizer, data_collator=data_collator,
)

print("Starting SFT training using chat template logic...")
trainer.train()

# ------------------ 8. 保存最终的LoRA适配器 (保持不变) ------------------
print("Saving final LoRA adapter...")
trainer.save_model(os.path.join(OUTPUT_DIR, "final_adapter"))
