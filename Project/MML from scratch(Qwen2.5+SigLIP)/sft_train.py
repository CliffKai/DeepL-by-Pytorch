from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoConfig
from transformers import Trainer, TrainingArguments
from PIL import Image
import torch
import os
import json
import argparse
from torch.utils.data import Dataset
from typing import List, Dict, Any
from train import VLMConfig, VLM


# ============ Step 2.1: find_assistant_tokens ============
def find_assistant_tokens(tokenizer, target):
    """在token序列中找到所有assistant回答的(start, end)区间"""
    result = []
    start_index = 0
    end_index = 0
    while start_index <= len(target) - 1:
        if target[start_index] != tokenizer('assistant')['input_ids'][0]:
            start_index += 1
            end_index += 1
        else:
            end_index += 1
            if target[end_index] == tokenizer('<|im_end|>')['input_ids'][0]:
                result.append((start_index + 1, end_index + 1))
                start_index = end_index + 1
    return result


# ============ Step 2.2: SFT Dataset ============
class SFTDataset(Dataset):
    def __init__(self, images_path, data_path, tokenizer, processor, config):
        super().__init__()
        self.images_path = images_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        with open(data_path, 'r', encoding='utf-8') as f:
            self.datas = json.load(f)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        sample = self.datas[index]
        try:
            image_name = 'COCO_train2014_' + str(sample['image'])
            conversations = sample['conversations']

            # 多轮对话转成 messages 格式
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            for conversation in conversations:
                if conversation['from'] == 'human':
                    messages.append({"role": "user", "content": conversation['value']})
                else:
                    messages.append({"role": "assistant", "content": conversation['value']})

            # 一次性格式化整段对话，替换<image>为占位符
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False
            ).replace('<image>', '<|image_pad|>' * self.config.image_pad_num)

            # tokenize 后找到所有 assistant 回答的区间
            input_ids = self.tokenizer(text)['input_ids']
            indexs = find_assistant_tokens(self.tokenizer, input_ids)

            # labels 全填 pad，只在 assistant 区间填回真实 token
            labels = len(input_ids) * [self.tokenizer.pad_token_id]
            for idx in indexs:
                labels[idx[0]:idx[1]] = input_ids[idx[0]:idx[1]]

            # 错位
            input_ids = input_ids[:-1]
            labels = labels[1:]

            # 加载图片
            image = Image.open(os.path.join(self.images_path, image_name)).convert('RGB')
            pixel_values = self.processor(text=None, images=image, return_tensors='pt')['pixel_values']

        except:
            # 兜底
            default_image = Image.new('RGB', (224, 224), color='white')
            pixel_values = self.processor(text=None, images=default_image, return_tensors='pt')['pixel_values']
            q_text = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": "You are a helpful assistant."},
                 {"role": "user", "content": "图片内容是什么\n<image>"}],
                tokenize=False,
                add_generation_prompt=True
            ).replace('<image>', '<|image_pad|>' * self.config.image_pad_num)
            a_text = '图片内容为空' + self.tokenizer.eos_token
            q_input_ids = self.tokenizer(q_text)['input_ids']
            a_input_ids = self.tokenizer(a_text)['input_ids']
            input_ids = q_input_ids + a_input_ids
            labels = [self.tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids
            input_ids = input_ids[:-1]
            labels = labels[1:]

        return {
            'input_ids': input_ids,
            'labels': labels,
            'pixel_values': pixel_values
        }


# ============ Step 2.3: Data Collator (same as train.py) ============
class MyDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(feature['input_ids']) for feature in features)
        input_ids = []
        labels = []
        pixel_values = []
        for feature in features:
            input_ids.append(feature['input_ids'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['input_ids'])))
            labels.append(feature['labels'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['labels'])))
            pixel_values.append(feature['pixel_values'])

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'pixel_values': torch.cat(pixel_values, dim=0)
        }


# ============ Step 2.4: SFT Training Script ============
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_model_path', type=str, required=True, help='预训练模型checkpoint路径')
    parser.add_argument('--images_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='save/sft')
    args = parser.parse_args()

    # 注册自定义模型，让AutoModelForCausalLM能识别VLM
    AutoConfig.register("vlm_model", VLMConfig)
    AutoModelForCausalLM.register(VLMConfig, VLM)

    # 加载预训练好的模型
    config = VLMConfig.from_pretrained(args.pretrain_model_path)
    model = VLM(config)
    from safetensors.torch import load_file
    state_dict = load_file(os.path.join(args.pretrain_model_path, 'model.safetensors'))
    model.load_state_dict(state_dict, strict=False)

    # SFT冻结策略: 冻结vision+linear, 解冻LLM
    # [BUG FIX] 原始代码为 if 'linear' in name or 'vision_model':
    #           'vision_model' 非空字符串恒为True，逻辑有误，修正为 'vision_model' in name
    for name, param in model.named_parameters():
        if 'linear' in name or 'vision_model' in name:
            param.requires_grad = False
        if 'llm_model' in name:
            param.requires_grad = True

    print(f'模型参数量为：{sum(p.numel() for p in model.parameters())}')
    print(f'模型可训练参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path)
    processor = AutoProcessor.from_pretrained(config.vision_model_path)

    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        per_device_train_batch_size=2,
        learning_rate=1e-4,
        num_train_epochs=5,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        gradient_accumulation_steps=8,
        logging_steps=100,
        report_to='tensorboard',
        dataloader_pin_memory=True,
        dataloader_num_workers=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=SFTDataset(args.images_path, args.data_path, tokenizer, processor, config),
        data_collator=MyDataCollator(tokenizer)
    )

    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(args.output_dir)
    trainer.save_state()
