import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config, GenerationConfig
import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate text with trained GPT2 model")
    parser.add_argument("--tokenizer_dir", type=str, required=True, help="tokenizer directory")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to saved model checkpoint")
    parser.add_argument("--prompt", type=str, default="Hello, world!", help="Prompt text")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--top_k", type=float, default=50, help="Top-k sampling")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on")
    args = parser.parse_args()

    # -------------------- Load tokenizer --------------------
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    # -------------------- Load model --------------------
    model = GPT2LMHeadModel.from_pretrained(args.checkpoint_dir)
    model.to(args.device)
    model.eval()

    # -------------------- GenerationConfig --------------------
    # 如果 checkpoint 里有 generation_config.json，HF 会自动加载
    # 否则可以自定义
    gen_config = model.generation_config
    gen_config.max_new_tokens = args.max_new_tokens
    gen_config.temperature = args.temperature
    gen_config.top_p = args.top_p
    gen_config.pad_token_id = tokenizer.pad_token_id
    gen_config.eos_token_id = tokenizer.eos_token_id

    # -------------------- Prepare inputs --------------------
    # 可以支持 batch
    prompts = [args.prompt] * args.batch_size
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    inputs = {k: v.to(args.device) for k, v in inputs.items()}

    # -------------------- Generate --------------------
    with torch.no_grad():
        outputs = model.generate(**inputs,
                                 max_new_tokens=gen_config.max_new_tokens,
                                 temperature=gen_config.temperature,
                                 top_p=gen_config.top_p,
                                 top_k=gen_config.top_k,
                                 pad_token_id=gen_config.pad_token_id,
                                 eos_token_id=gen_config.eos_token_id,
                                 bos_token_id=gen_config.bos_token_id)

    # -------------------- Decode and print --------------------
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    for i, text in enumerate(decoded):
        print(f"=== Output {i} ===")
        print(text)
        print("=================\n")


if __name__ == "__main__":
    main()
