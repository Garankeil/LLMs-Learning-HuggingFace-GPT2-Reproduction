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


# ------------------ 4. 数据集处理 (最终修正版) ------------------
class SFTDataset(Dataset):
    def __init__(self, file_path: str, tokenizer_d, max_length: int = 512):
        self.tokenizer = tokenizer_d
        self.max_length = max_length
        print(f"Loading data from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]
        print(f"Loaded {len(self.data)} examples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        conversations = item['conversations']

        # 第一步：一次性将整个对话历史应用模板，生成完整的 input_ids
        # 这是确保与模板逻辑完全一致的唯一正确方法
        full_input_text = self.tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=False  # 在训练时我们不需要末尾的 assistant 提示
        )
        input_ids = self.tokenizer.encode(full_input_text, add_special_tokens=False)
        
        # 第二步：初始化 labels，默认所有 token 都不计算损失
        labels = [-100] * len(input_ids)

        # 第三步：精确定位并标记出 assistant 的回答部分
        # 我们通过独立编码 assistant 的内容，然后在完整的 input_ids 中查找它
        current_search_position = 0
        for turn in conversations:
            if turn['role'] == 'assistant':
                # 只编码内容，不加任何特殊 token
                assistant_content_ids = self.tokenizer.encode(turn['content'], add_special_tokens=False)
                
                # 在 input_ids 中查找这段 content_ids 的起始位置
                # 我们从上一次找到的位置之后开始搜索，以处理多轮对话
                try:
                    start_index = input_ids.index(assistant_content_ids[0], current_search_position)
                    
                    # 验证这是否是一个真正的匹配
                    if input_ids[start_index : start_index + len(assistant_content_ids)] == assistant_content_ids:
                        # 找到了！现在我们用真实的 token_id 替换掉 labels 中对应位置的 -100
                        labels[start_index : start_index + len(assistant_content_ids)] = assistant_content_ids
                        
                        # 更新下一次搜索的起始位置
                        current_search_position = start_index + len(assistant_content_ids)

                except ValueError:
                    # 如果在 input_ids 中找不到 assistant 的第一个 token，说明数据可能有问题
                    # 在这里可以添加一些日志或错误处理
                    # print(f"Warning: Could not find assistant content in input_ids for item {idx}")
                    pass

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
