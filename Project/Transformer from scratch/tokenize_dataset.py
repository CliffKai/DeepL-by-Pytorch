# 将文本数据集转化为模型可以处理的.npy 数字格式

import numpy as np
from src.bpe.tokenizer import Tokenizer

# --- 配置 ---
TOKENIZER_DIR = "./bpe_tokenizer"
DATA_DIR = "./data"
OUTPUT_DIR = "./data" # 直接保存在数据目录下

# 初始化分词器
tokenizer = Tokenizer.from_files(
    vocab_filepath=f"{TOKENIZER_DIR}/tinystories_vocab.json",
    merges_filepath=f"{TOKENIZER_DIR}/tinystories_merges.txt",
    special_tokens=["<|endoftext|>"]
)

def tokenize_file(input_path, output_path):
    print(f"正在处理: {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    ids = tokenizer.encode(text)
    # 使用 uint16 格式保存，节省空间
    arr = np.array(ids, dtype=np.uint16)

    np.save(output_path, arr)
    print(f"Tokenized 数据已保存至: {output_path} (shape: {arr.shape}, dtype: {arr.dtype})")

# 处理训练集和验证集
tokenize_file(
    f"{DATA_DIR}/TinyStoriesV2-GPT4-train.txt",
    f"{OUTPUT_DIR}/tinystories_train.npy"
)
tokenize_file(
    f"{DATA_DIR}/TinyStoriesV2-GPT4-valid.txt",
    f"{OUTPUT_DIR}/tinystories_val.npy"
)
print("所有文件处理完毕！")