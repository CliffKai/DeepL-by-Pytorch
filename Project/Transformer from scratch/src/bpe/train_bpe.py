import os
import regex as re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

# GPT-2 使用的预分词正则表达式模式
# 这个模式能够处理大多数情况，包括撇号、单词、数字、标点和空格
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
COMPILED_PAT = re.compile(PAT)


def get_stats(word_counts: dict[tuple[bytes, ...], int]) -> Counter:
    """
    计算语料库中所有相邻token对的频率。

    Args:
        word_counts (dict): 一个字典，键是表示单词的字节元组，值是该单词的频率。

    Returns:
        Counter: 一个计数器，存储了每个字节对及其出现的总次数。
    """
    pair_counts = Counter()
    for word_tuple, count in word_counts.items():
        for i in range(len(word_tuple) - 1):
            pair = (word_tuple[i], word_tuple[i + 1])
            pair_counts[pair] += count
    return pair_counts


def merge(
    word_counts: dict[tuple[bytes, ...], int],
    pair_to_merge: tuple[bytes, bytes],
    new_token: bytes,
) -> dict[tuple[bytes, ...], int]:
    """
    在一个“单词”词汇表中执行一次合并操作。

    Args:
        word_counts (dict): 当前的单词及其频率。
        pair_to_merge (tuple): 需要被合并的字节对。
        new_token (bytes): 由被合并的字节对创建的新token。

    Returns:
        dict: 合并操作后更新的单词及其频率。
    """
    new_word_counts = defaultdict(int)
    p1, p2 = pair_to_merge
    for word_tuple, count in word_counts.items():
        i = 0
        new_word_tuple = []
        while i < len(word_tuple):
            # 查找并替换需要合并的字节对
            if i < len(word_tuple) - 1 and word_tuple[i] == p1 and word_tuple[i + 1] == p2:
                new_word_tuple.append(new_token)
                i += 2
            else:
                new_word_tuple.append(word_tuple[i])
                i += 1
        new_word_counts[tuple(new_word_tuple)] += count
    return dict(new_word_counts)


def parallel_pre_tokenize(text: str, special_tokens: list[str], n_workers: Optional[int] = None) -> Counter:
    """
    使用多进程并行地对文本进行预分词和计数。

    Args:
        text (str): 完整的输入文本。
        special_tokens (list[str]): 特殊token列表。
        n_workers (int, optional): 使用的进程数。默认为 os.cpu_count()。

    Returns:
        Counter: 每个预分词块（word）及其频率的计数器。
    """
    if not text:
        return Counter()
    
    text_chunks_to_process = []
    if special_tokens:
        # 创建用于分割的正则表达式，并使用捕获组来保留分隔符
        special_pattern = f"({'|'.join(re.escape(st) for st in special_tokens)})"
        all_chunks = re.split(special_pattern, text)
        
        # re.split 会将文本和分隔符交替放入列表。
        # 偶数索引是文本块，奇数索引是特殊token本身。
        # 我们只对文本块进行预分词。
        for i in range(0, len(all_chunks), 2):
            chunk = all_chunks[i]
            if chunk:  # 仅处理非空文本块
                text_chunks_to_process.append(chunk)
    else:
        # 如果没有特殊token，整个文本就是一个块
        text_chunks_to_process.append(text)

    word_counts = Counter()

    # 使用进程池并行处理每个文本块
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # map 会自动处理块的分发和结果的收集
        results = executor.map(worker_tokenize_chunk, text_chunks_to_process)
        
        # 聚合所有进程的结果
        for result_counter in results:
            word_counts.update(result_counter)
            
    return word_counts

def worker_tokenize_chunk(chunk: str) -> Counter:
    """
    单个工作进程执行的函数，对一个文本块进行预分词和计数。
    """
    counts = Counter()
    # 使用 finditer 以提高内存效率
    for match in COMPILED_PAT.finditer(chunk):
        word_bytes = match.group(0).encode("utf-8")
        counts[word_bytes] += 1
    return counts


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    在给定的输入语料库上训练一个BPE分词器。

    Args:
        input_path (str | os.PathLike): BPE训练数据的路径。
        vocab_size (int): 最终词汇表的大小（包括初始字节和特殊token）。
        special_tokens (list[str]): 特殊token列表。

    Returns:
        一个元组，包含:
        - vocab (dict[int, bytes]): 训练好的词汇表 (token_id -> bytes)。
        - merges (list[tuple[bytes, bytes]]): 按顺序学习到的BPE合并规则。
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if vocab_size < 256:
        raise ValueError("Vocab size must be at least 256.")

    # 1. 初始化词汇表
    # 词汇表从 256 个基本字节开始
    vocab_list = [bytes([i]) for i in range(256)]
    # 添加特殊token到词汇表
    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        if token_bytes not in vocab_list:
            vocab_list.append(token_bytes)

    # 2. 预分词 (内存高效的分块处理方式)
    print("  - 开始分块预分词，以节省内存...")
    pre_token_counts = Counter()
    # 每次处理 100MB 的文件块
    chunk_size = 100 * 1024 * 1024 

    with open(input_path, "r", encoding="utf-8") as f:
        while True:
            chunk_text = f.read(chunk_size)
            if not chunk_text:
                break
            
            # 使用并行函数处理当前块
            chunk_counts = parallel_pre_tokenize(chunk_text, special_tokens)
            # 聚合结果
            pre_token_counts.update(chunk_counts)
            print(f"  - 已处理一个数据块，当前总词块数: {len(pre_token_counts)}")

    print("  - 所有数据块预分词完成。")

    # 将单词转换为字节元组的表示形式
    word_counts = {
        tuple(bytes([b]) for b in word): count
        for word, count in pre_token_counts.items()
    }

    # 3. 迭代合并
    merges = []
    num_merges = vocab_size - len(vocab_list)

    for i in range(num_merges):
        # 计算当前所有相邻字节对的频率
        stats = get_stats(word_counts)
        if not stats:
            # 如果没有可合并的对，则提前停止
            break
        
        # 找到频率最高的字节对。如果频率相同，max()会根据元组的字典序来决定，
        best_pair = max(stats, key=lambda p: (stats[p], p))
        
        # 执行合并
        new_token = best_pair[0] + best_pair[1]
        word_counts = merge(word_counts, best_pair, new_token)
        
        # 记录这次合并
        merges.append(best_pair)
        if new_token not in vocab_list:
            vocab_list.append(new_token)
    
    # 4. 构建最终的词汇表字典
    final_vocab = {i: token_bytes for i, token_bytes in enumerate(vocab_list)}

    return final_vocab, merges