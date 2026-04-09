from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoConfig
from PIL import Image
import torch
import torch.nn.functional as F
import argparse
from train import VLMConfig, VLM


# ============ Step 5.1: Multi-Image Inference ============
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help='多图SFT模型checkpoint路径')
parser.add_argument('--image_paths', type=str, nargs='+', required=True, help='测试图片路径列表')
parser.add_argument('--prompt', type=str, default='描述这些图片的内容')
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()

# 注册并加载模型
AutoConfig.register("vlm_model", VLMConfig)
AutoModelForCausalLM.register(VLMConfig, VLM)

model = AutoModelForCausalLM.from_pretrained(args.model_path)
model.to(args.device)
model.eval()

config = VLMConfig.from_pretrained(args.model_path)
tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path)
processor = AutoProcessor.from_pretrained(config.vision_model_path)

# 准备文本: 每张图对应一个<image>标记
image_tags = '<image>' * len(args.image_paths)
q_text = tokenizer.apply_chat_template(
    [{"role": "system", "content": "You are a helpful assistant."},
     {"role": "user", "content": f'{args.prompt}\n{image_tags}'}],
    tokenize=False,
    add_generation_prompt=True
).replace('<image>', '<|image_pad|>' * 49 + '\n')

input_ids = tokenizer(q_text, return_tensors='pt')['input_ids']
input_ids = input_ids.to(args.device)

# 准备多张图片
images = [Image.open(p).convert("RGB") for p in args.image_paths]
pixel_values = processor(text=None, images=images).pixel_values  # (num_images, 3, 224, 224)
pixel_values = pixel_values.to(args.device)

# 自回归生成 (同 test.py)
max_new_tokens = 100
temperature = 0.0
top_k = None
eos = tokenizer.eos_token_id
s = input_ids.shape[1]

while input_ids.shape[1] < s + max_new_tokens - 1:
    inference_res = model(input_ids, None, pixel_values)
    logits = inference_res.logits
    logits = logits[:, -1, :]

    # [BUG] 原始代码除以1.0，无任何效果，疑似想实现重复惩罚(repetition penalty)
    # 正确做法应除以大于1的值（如1.2）来降低已出现token的概率
    for token in set(input_ids.tolist()[0]):
        logits[:, token] /= 1.0

    if temperature == 0.0:
        _, idx_next = torch.topk(logits, k=1, dim=-1)
    else:
        logits = logits / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

    if idx_next == eos:
        break

    input_ids = torch.cat((input_ids, idx_next), dim=1)

print(tokenizer.decode(input_ids[:, s:][0]))
