import argparse
import torch
import os

from src.model.transformer import TransformerLM
from src.bpe.tokenizer import Tokenizer
from src.io import load_checkpoint
from src.nn_utils import softmax

def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Generate text from a trained Transformer model.")
    
    # 模型和分词器路径
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint (.pt file)")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to the vocabulary file (.json)")
    parser.add_argument("--merges_path", type=str, required=True, help="Path to the merges file (.txt)")
    parser.add_argument("--special_tokens", type=str, nargs='*', default=["<|endoftext|>"], help="List of special tokens")

    # 生成参数
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="The starting prompt for generation")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling. 1.0 = no change, < 1.0 = less random, > 1.0 = more random.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling threshold. 0.0 disables it.")

    # 性能
    parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cpu', 'cuda', 'mps')")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for faster generation")

    return parser.parse_args()

@torch.no_grad()
def generate(model, tokenizer, prompt_ids, max_new_tokens, temperature, top_p):
    """生成文本的核心函数"""
    model.eval()
    
    # 打印初始提示
    print(tokenizer.decode(prompt_ids.tolist()[0]), end="", flush=True)

    ids = prompt_ids
    for _ in range(max_new_tokens):
        # 保证上下文长度不超过模型限制
        context = ids[:, -model.token_embeddings.weight.shape[1]:]

        # 前向传播
        logits = model(context)
        # 只关心最后一个时间步的 logits
        logits = logits[:, -1, :]

        # 应用温度
        if temperature > 0:
            logits = logits / temperature

        # 计算概率
        probs = softmax(logits, dim=-1)

        # 应用 Top-p (Nucleus) 采样
        if 0 < top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # 找到累积概率超过 p 的位置
            sorted_indices_to_remove = cumulative_probs > top_p
            # 至少保留一个 token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # 创建一个 mask 将低概率的 token 置为0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            # [重要修复] 将 indices_to_remove 从 1D 变形为 2D 以匹配 probs 的维度
            indices_to_remove = indices_to_remove.unsqueeze(0)
            probs.scatter_(1, indices_to_remove, 0)
            # 重新归一化
            probs = probs / probs.sum(dim=-1, keepdim=True)
        
        # 从分布中采样
        next_token_id = torch.multinomial(probs, num_samples=1)
        
        # 检查是否是结束符
        if next_token_id.item() in tokenizer.special_tokens_encoder.values():
            break

        # 解码并打印
        print(tokenizer.decode(next_token_id.tolist()[0]), end="", flush=True)

        # 将新生成的 token 添加到序列中，为下一步做准备
        ids = torch.cat((ids, next_token_id), dim=1)

    print("\n--- Generation finished ---")


def main():
    args = get_args()
    
    # 设置设备
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    torch.manual_seed(1337)
    
    # 加载分词器
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab_path,
        merges_filepath=args.merges_path,
        special_tokens=args.special_tokens
    )

    # 加载模型
    print(f"Loading model from {args.checkpoint_path}...")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    # 从检查点中获取模型参数来初始化模型
    # 注意：需要确保这些参数与训练时使用的参数一致
    # 更好的做法是将模型配置保存在检查点中
    model_args = checkpoint.get("model_args", None)
    if model_args is None:
        # 如果检查点中没有保存模型参数，就从命令行参数中推断
        # 这是一个简化的假设，实际项目中应将配置与模型一起保存
        print("Warning: model args not found in checkpoint. Inferring from weights.")
        d_model = checkpoint['model_state']['token_embeddings.weight'].shape[1]
        vocab_size = checkpoint['model_state']['token_embeddings.weight'].shape[0]
        # 其他参数可能需要手动设置或从 state_dict 的键中推断
        num_layers = max([int(k.split('.')[1]) for k in checkpoint['model_state'] if k.startswith('layers.')]) + 1
        model_args = {
            "vocab_size": vocab_size, "d_model": d_model, "num_layers": num_layers,
            "context_length": 256, "num_heads": 16, "d_ff": 1344, "rope_theta": 10000.0
        }

    model = TransformerLM(**model_args, device=device)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)

    if args.compile:
        print("Compiling model for generation...")
        model = torch.compile(model)

    # 编码初始提示
    prompt_ids = tokenizer.encode(args.prompt)
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

    # 生成文本
    generate(model, tokenizer, prompt_tensor, args.max_new_tokens, args.temperature, args.top_p)

if __name__ == "__main__":
    main()