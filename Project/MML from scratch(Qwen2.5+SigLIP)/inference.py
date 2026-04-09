from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoConfig
from PIL import Image
import torch
import torch.nn.functional as F
import argparse
from train import VLMConfig, VLM


# ============ Step 4.1: Model Loading ============
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help='SFT模型checkpoint路径')
parser.add_argument('--image_path', type=str, required=True, help='测试图片路径')
parser.add_argument('--prompt', type=str, default='描述图片内容')
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()

# 注册自定义模型
AutoConfig.register("vlm_model", VLMConfig)
AutoModelForCausalLM.register(VLMConfig, VLM)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(args.model_path)
model.to(args.device)
model.eval()

# 获取 tokenizer 和 processor
config = VLMConfig.from_pretrained(args.model_path)
tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path)
processor = AutoProcessor.from_pretrained(config.vision_model_path)

# 准备文本输入: 套用chat模板，替换<image>为49个占位符
q_text = tokenizer.apply_chat_template(
    [{"role": "system", "content": "You are a helpful assistant."},
     {"role": "user", "content": f'{args.prompt}\n<image>'}],
    tokenize=False,
    add_generation_prompt=True
).replace('<image>', '<|image_pad|>' * 49)

input_ids = tokenizer(q_text, return_tensors='pt')['input_ids']
input_ids = input_ids.to(args.device)

# 准备图片输入
image = Image.open(args.image_path).convert("RGB")
pixel_values = processor(text=None, images=image).pixel_values
pixel_values = pixel_values.to(args.device)


# ============ Step 4.2: Auto-regressive Generation Loop ============
max_new_tokens = 100
temperature = 0.0    # 0.0 = 贪心解码(每次取最大概率)
top_k = None
eos = tokenizer.eos_token_id
s = input_ids.shape[1]   # 记录原始输入长度，后面只解码新生成的部分

while input_ids.shape[1] < s + max_new_tokens - 1:
    # 前向传播，labels=None不算loss
    inference_res = model(input_ids, None, pixel_values)
    logits = inference_res.logits       # (1, seq_len, vocab_size)
    logits = logits[:, -1, :]           # (1, vocab_size) 只取最后一个位置

    # [BUG] 原始代码除以1.0，无任何效果，疑似想实现重复惩罚(repetition penalty)
    # 正确做法应除以大于1的值（如1.2）来降低已出现token的概率
    for token in set(input_ids.tolist()[0]):
        logits[:, token] /= 1.0

    if temperature == 0.0:
        # 贪心: 直接取最大分数的token
        _, idx_next = torch.topk(logits, k=1, dim=-1)
    else:
        # 温度采样 + top-k
        logits = logits / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

    # 遇到eos停止
    if idx_next == eos:
        break

    # 拼接新token到序列末尾
    input_ids = torch.cat((input_ids, idx_next), dim=1)

# 解码输出: 只取新生成的部分
print(tokenizer.decode(input_ids[:, s:][0]))
