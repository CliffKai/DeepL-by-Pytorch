import time
import json
import base64
from src.bpe.train_bpe import run_train_bpe

# --- 配置 ---
# 输入的文本文件
INPUT_TXT_PATH = "/root/data/cs336/TinyStoriesV2-GPT4-train.txt" 
# 输出目录
OUTPUT_DIR = "./bpe_tokenizer"
# 词汇表大小 (PDF为TinyStories推荐10000)
VOCAB_SIZE = 10000
# 特殊token
SPECIAL_TOKENS = ["<|endoftext|>"]

print("开始训练 BPE Tokenizer...")
start_time = time.time()

# 调用你的 BPE 训练函数
vocab, merges = run_train_bpe(
    input_path=INPUT_TXT_PATH,
    vocab_size=VOCAB_SIZE,
    special_tokens=SPECIAL_TOKENS
)

end_time = time.time()
print(f"BPE 训练完成！耗时: {end_time - start_time:.2f} 秒")

# --- 保存文件 ---
# 1. 保存词汇表
#    为了兼容 from_files，我们将 bytes 转换为 str
vocab_for_json = {k: base64.b64encode(v).decode('ascii') for k, v in vocab.items()}
vocab_filepath = f"{OUTPUT_DIR}/tinystories_vocab.json"
with open(vocab_filepath, "w", encoding="utf-8") as f:
    json.dump(vocab_for_json, f, ensure_ascii=False, indent=2)
print(f"词汇表已保存至: {vocab_filepath}")

# 2. 保存合并规则
merges_filepath = f"{OUTPUT_DIR}/tinystories_merges.txt"
with open(merges_filepath, "w", encoding="utf-8") as f:
    for p1, p2 in merges:
        f.write(f"{p1.decode('utf-8', 'replace')}\t{p2.decode('utf-8', 'replace')}\n")
print(f"合并规则已保存至: {merges_filepath}")