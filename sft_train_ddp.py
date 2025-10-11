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


# ------------------ 4. 数据集处理 (最终的、基于边界的版本) ------------------
class SFTDataset(Dataset):
    def __init__(self, file_path: str, tokenizer_d, max_length: int = 512):
        self.tokenizer = tokenizer_d
        self.max_length = max_length
        
        # 为了高效，我们在初始化时就获取边界标记的 token ID
        # assistant 回答的开始，是由 user 回合的模板触发的
        # 我们只取这个提示的最后一个 token 作为开始的信号
        self.assistant_prompt_end_token_id = self.tokenizer.encode(
            '<|im_start|>assistant\n', add_special_tokens=False
        )[-1]
        
        # assistant 回答的结束标记
        self.im_end_token_id = self.tokenizer.eos_token_id # 根据您的 tokenizer，<|im_end|> 就是 eos_token

        print(f"Loading data from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]
        print(f"Loaded {len(self.data)} examples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        conversations = item['conversations']

        # 第一步：一次性生成完整的 input_ids
        input_ids = self.tokenizer.apply_chat_template(
            conversations,
            tokenize=True, # 直接让模板返回 token IDs
            add_special_tokens=False,
            add_generation_prompt=False,
        )
        
        # 第二步：初始化 labels，默认所有 token 都不计算损失
        labels = [-100] * len(input_ids)
        
        # 第三步：基于边界标记，找到所有 assistant 的回答并标记 labels
        current_position = 0
        while current_position < len(input_ids):
            try:
                # 1. 寻找 assistant 回答的“开始”边界
                # 我们寻找 user 回合末尾的那个 assistant 提示符
                start_index = input_ids.index(self.assistant_prompt_end_token_id, current_position)
                
                # assistant 的实际内容从这个提示符之后开始
                content_start_index = start_index + 1

                # 2. 寻找 assistant 回答的“结束”边界
                # 从内容开始的位置，寻找第一个 <|im_end|>
                end_index = input_ids.index(self.im_end_token_id, content_start_index)
                
                # 3. 标记证据
                # 我们要学习的，就是从 content_start_index 到 end_index (包含) 的所有内容
                labels[content_start_index : end_index + 1] = input_ids[content_start_index : end_index + 1]
                
                # 4. 更新下次搜寻的起点
                current_position = end_index + 1
                
            except ValueError:
                # 如果找不到更多的 assistant 回答边界，就跳出循环
                break

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
