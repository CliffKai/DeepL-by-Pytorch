import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoConfig
from PIL import Image
import torch
import torch.nn.functional as F
import argparse
from train import VLMConfig, VLM


# ============ Step 6.1: Gradio Demo ============
parser = argparse.ArgumentParser()
parser.add_argument('--pretrain_model_path', type=str, required=True, help='预训练模型路径')
parser.add_argument('--sft_model_path', type=str, required=True, help='SFT模型路径')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--port', type=int, default=7891)
args = parser.parse_args()

# 注册自定义模型
AutoConfig.register("vlm_model", VLMConfig)
AutoModelForCausalLM.register(VLMConfig, VLM)

# 加载两个模型用于对比
pretrain_model = AutoModelForCausalLM.from_pretrained(args.pretrain_model_path)
pretrain_model.to(args.device)
pretrain_model.eval()

sft_model = AutoModelForCausalLM.from_pretrained(args.sft_model_path)
sft_model.to(args.device)
sft_model.eval()

config = VLMConfig.from_pretrained(args.sft_model_path)
tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path)
processor = AutoProcessor.from_pretrained(config.vision_model_path)


def generate(mode, image_input, text_input, max_new_tokens=100, temperature=0.0, top_k=None):
    # 准备文本
    q_text = tokenizer.apply_chat_template(
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": f'{text_input}\n<image>'}],
        tokenize=False,
        add_generation_prompt=True
    ).replace('<image>', '<|image_pad|>' * 49)

    input_ids = tokenizer(q_text, return_tensors='pt')['input_ids']
    input_ids = input_ids.to(args.device)

    # 准备图片
    pixel_values = processor(text=None, images=image_input).pixel_values
    pixel_values = pixel_values.to(args.device)

    eos = tokenizer.eos_token_id
    s = input_ids.shape[1]

    # 选择模型
    model = pretrain_model if mode == 'pretrain' else sft_model

    # 自回归生成
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

    return tokenizer.decode(input_ids[:, s:][0])


# Gradio 界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="选择图片")
        with gr.Column(scale=1):
            mode = gr.Radio(["pretrain", "sft"], label="选择模型")
            text_input = gr.Textbox(label="输入文本")
            text_output = gr.Textbox(label="输出文本")
            generate_button = gr.Button("生成")
            generate_button.click(generate, inputs=[mode, image_input, text_input], outputs=text_output)

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=args.port)
