# inference_gpt2.py

import torch
from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer
import argparse

def load_model_and_tokenizer(model_path, tokenizer_path):
    """加载模型和tokenizer"""
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型配置
    config = GPT2Config.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_config(config)
    
    # 加载训练好的权重
    model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin", map_location='cpu'))
    model.eval()
    
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0, top_k=50, top_p=0.95, do_sample=True):
    """
    生成文本
    
    Args:
        model: 训练好的GPT2模型
        tokenizer: 对应的tokenizer
        prompt: 输入的提示文本
        max_length: 生成的最大长度
        temperature: 温度参数，控制随机性
        top_k: Top-K采样
        top_p: Top-P(Nucleus)采样
        do_sample: 是否采样，False则使用贪婪解码
    """
    # 编码输入文本
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    with torch.no_grad():
        # 生成文本
        output = model.generate(
            input_ids,
            max_length=max_length + input_ids.size(1),  # 总长度 = prompt长度 + 生成长度
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
    
    # 解码生成的文本
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def interactive_generation(model, tokenizer, max_length=100):
    """交互式文本生成"""
    print("GPT2 文本生成器启动！输入 'quit' 退出。")
    print(f"模型最大位置长度: {model.config.n_positions}")
    print("-" * 50)
    
    while True:
        prompt = input("\n请输入提示文本: ")
        if prompt.lower() == 'quit':
            break
            
        try:
            generated = generate_text(model, tokenizer, prompt, max_length=max_length)
            print(f"\n生成结果:\n{generated}")
            
            # 只显示生成的部分（去除prompt）
            if generated.startswith(prompt):
                generated_part = generated[len(prompt):].strip()
                print(f"\n仅生成部分:\n{generated_part}")
        except Exception as e:
            print(f"生成过程中出现错误: {e}")

def batch_generation(model, tokenizer, prompts, max_length=100):
    """批量生成文本"""
    results = []
    for i, prompt in enumerate(prompts):
        print(f"处理第 {i+1}/{len(prompts)} 个提示...")
        try:
            generated = generate_text(model, tokenizer, prompt, max_length=max_length)
            results.append({
                'prompt': prompt,
                'generated': generated,
                'generated_only': generated[len(prompt):].strip() if generated.startswith(prompt) else generated
            })
        except Exception as e:
            print(f"处理提示 '{prompt}' 时出错: {e}")
            results.append({
                'prompt': prompt,
                'generated': '',
                'generated_only': '',
                'error': str(e)
            })
    return results

def main():
    parser = argparse.ArgumentParser(description='GPT2 推理脚本')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='训练好的模型路径')
    parser.add_argument('--tokenizer_path', type=str, required=True, 
                       help='tokenizer路径')
    parser.add_argument('--mode', type=str, choices=['interactive', 'single', 'batch'], 
                       default='interactive', help='推理模式')
    parser.add_argument('--prompt', type=str, default='', 
                       help='单次推理的提示文本')
    parser.add_argument('--max_length', type=int, default=100, 
                       help='生成的最大长度')
    parser.add_argument('--temperature', type=float, default=1.0, 
                       help='温度参数')
    parser.add_argument('--top_k', type=int, default=50, 
                       help='Top-K采样参数')
    parser.add_argument('--top_p', type=float, default=0.95, 
                       help='Top-P采样参数')
    
    args = parser.parse_args()
    
    # 检查CUDA可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型和tokenizer
    print("正在加载模型...")
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.tokenizer_path)
    model = model.to(device)
    print(f"模型加载完成！最大位置长度: {model.config.n_positions}")
    
    if args.mode == 'interactive':
        # 交互式生成
        interactive_generation(model, tokenizer, args.max_length)
        
    elif args.mode == 'single':
        # 单次生成
        if not args.prompt:
            print("错误: 单次生成模式需要提供 --prompt 参数")
            return
        
        generated = generate_text(
            model, tokenizer, args.prompt, 
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        print(f"提示: {args.prompt}")
        print(f"生成: {generated}")
        if generated.startswith(args.prompt):
            print(f"仅生成部分: {generated[len(args.prompt):].strip()}")
    
    elif args.mode == 'batch':
        # 批量生成示例
        prompts = [
            "今天天气很好",
            "人工智能是",
            "机器学习",
            "深度学习"
        ]
        
        results = batch_generation(model, tokenizer, prompts, args.max_length)
        
        print("\n批量生成结果:")
        print("-" * 60)
        for i, result in enumerate(results):
            print(f"示例 {i+1}:")
            print(f"  提示: {result['prompt']}")
            print(f"  生成: {result['generated_only']}")
            print("-" * 60)

if __name__ == "__main__":
    main()
