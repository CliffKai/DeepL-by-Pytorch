import os
import time
import argparse
from contextlib import nullcontext

import numpy as np
import torch
import wandb

from src.model.transformer import TransformerLM
from src.optim_sched import get_adamw_cls, get_lr_cosine_schedule
from src.data import get_batch
from src.io import save_checkpoint, load_checkpoint
from src.nn_utils import cross_entropy, gradient_clipping

def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Train a Transformer Language Model")

    # I/O 参数
    parser.add_argument("--out_dir", type=str, default="checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to tokenized training data (.npy)")
    parser.add_argument("--val_data_path", type=str, required=True, help="Path to tokenized validation data (.npy)")
    parser.add_argument("--init_from", type=str, default="scratch", choices=["scratch", "resume"], help="Start from scratch or resume from out_dir")
    
    # WandB 日志
    parser.add_argument("--wandb_log", action="store_true", help="Enable logging to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="cs336_assignment1", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=f"train-{int(time.time())}", help="W&B run name")

    # 模型参数
    parser.add_argument("--vocab_size", type=int, required=True, help="Vocabulary size")
    parser.add_argument("--context_length", type=int, default=256, help="Maximum context length")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=1344, help="Dimension of the feed-forward layer")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="Theta for RoPE")

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--max_steps", type=int, default=20000, help="Total training steps")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Maximum learning rate")
    parser.add_argument("--min_learning_rate", type=float, default=3e-5, help="Minimum learning rate")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--beta2", type=float, default=0.95, help="AdamW beta2")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")

    # 评估与日志记录
    parser.add_argument("--eval_interval", type=int, default=250, help="Steps between evaluations")
    parser.add_argument("--eval_steps", type=int, default=100, help="Number of steps for evaluation")
    parser.add_argument("--log_interval", type=int, default=10, help="Steps between logging training loss")
    parser.add_argument("--save_interval", type=int, default=5000, help="Steps between saving checkpoints")
    
    # 性能
    parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cpu', 'cuda', 'mps')")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for performance")

    return parser.parse_args()

@torch.no_grad()
def evaluate(model, data, context_length, batch_size, device, max_steps):
    """在验证集上评估模型"""
    model.eval()
    losses = torch.zeros(max_steps)
    for k in range(max_steps):
        X, Y = get_batch(
            dataset=data,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )
        logits = model(X)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
        losses[k] = loss.item()
    model.train()
    return losses.mean()

def main():
    args = get_args()

    # 设置设备
    device = args.device
    if "cuda" in device and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    
    torch.manual_seed(1337)
    if "cuda" in device:
        torch.cuda.manual_seed(1337)

    # 加载数据
    print("Loading data...")
    # 使用 mmap_mode='r' 进行内存高效加载
    train_data = np.load(args.train_data_path, mmap_mode='r')
    val_data = np.load(args.val_data_path, mmap_mode='r')

    # 初始化模型
    model_args = {
        "vocab_size": args.vocab_size, "context_length": args.context_length,
        "d_model": args.d_model, "num_layers": args.num_layers,
        "num_heads": args.num_heads, "d_ff": args.d_ff,
        "rope_theta": args.rope_theta, "device": device,
    }
    model = TransformerLM(**model_args)
    model.to(device)

    # 编译模型以提高性能
    if args.compile:
        print("Compiling the model... (this may take a minute)")
        if "mps" in device:
            # MPS 后端支持有限
            model = torch.compile(model, backend="aot_eager")
        else:
            model = torch.compile(model)

    # 初始化优化器
    AdamW = get_adamw_cls()
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )
    
    # 检查点和续训逻辑
    start_step = 0
    if args.init_from == "resume":
        ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
        if os.path.exists(ckpt_path):
            print(f"Resuming training from {ckpt_path}")
            start_step = load_checkpoint(src=ckpt_path, model=model, optimizer=optimizer)
        else:
            print(f"Checkpoint not found at {ckpt_path}, starting from scratch.")
    
    os.makedirs(args.out_dir, exist_ok=True)

    # WandB 设置
    if args.wandb_log:
        wandb.init(
        project="cs336_assignment1",
        entity="kkwei0819-beijing-jiaotong-university",                 
        name=args.wandb_run_name,
        config=vars(args),             
    )
    
    # 训练循环
    print(f"Starting training for {args.max_steps} steps...")
    t0 = time.time()
    for step in range(start_step, args.max_steps):
        # 获取学习率
        lr = get_lr_cosine_schedule(
            it=step, max_learning_rate=args.learning_rate,
            min_learning_rate=args.min_learning_rate,
            warmup_iters=args.warmup_steps, cosine_cycle_iters=args.max_steps
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 评估
        if step % args.eval_interval == 0 and step > 0:
            val_loss = evaluate(model, val_data, args.context_length, args.batch_size, device, args.eval_steps)
            print(f"Step {step}: Val Loss = {val_loss:.4f}, Val PPL = {torch.exp(val_loss):.2f}")
            if args.wandb_log:
                wandb.log({
                    "eval/loss": val_loss,
                    "eval/perplexity": torch.exp(val_loss),
                }, step=step)
            
            # 保存检查点
            if step % args.save_interval == 0:
                ckpt_path = os.path.join(args.out_dir, f"ckpt_step{step}.pt")
                save_checkpoint(model=model, optimizer=optimizer, iteration=step, out=ckpt_path)
                print(f"Checkpoint saved to {ckpt_path}")
            # 始终保存最新检查点（用于续训）
            latest_path = os.path.join(args.out_dir, "ckpt.pt")
            save_checkpoint(model=model, optimizer=optimizer, iteration=step, out=latest_path)


        # 获取训练数据
        X, Y = get_batch(
            dataset=train_data,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=device,
        )

        # 前向和后向传播
        logits = model(X)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # 梯度裁剪
        if args.grad_clip > 0:
            gradient_clipping(model.parameters(), args.grad_clip)

        optimizer.step()

        # 日志记录
        if step % args.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            tokens_per_sec = (args.batch_size * args.context_length * args.log_interval) / dt
            print(f"Step {step:6d} | Loss: {loss.item():.4f} | LR: {lr:.2e} | Time: {dt*1000:.2f}ms | Tokens/sec: {tokens_per_sec:.0f}")
            if args.wandb_log:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": lr,
                    "perf/tokens_per_sec": tokens_per_sec,
                }, step=step)

    print("Training finished.")

if __name__ == "__main__":
    main()