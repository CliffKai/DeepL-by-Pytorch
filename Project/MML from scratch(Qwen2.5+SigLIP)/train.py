from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import AutoProcessor, AutoModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import Trainer, TrainingArguments
from PIL import Image
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import argparse
from torch.utils.data import Dataset
from typing import List, Dict, Any


# ============ Step 1.1: VLMConfig ============
class VLMConfig(PretrainedConfig):
    """VLM模型的配置类，存储所有超参数"""
    model_type = "vlm_model"

    def __init__(
        self,
        llm_model_path=None,
        vision_model_path=None,
        freeze_vision_model=True,
        image_pad_num=49,
        **kwargs
    ):
        self.llm_model_path = llm_model_path
        self.vision_model_path = vision_model_path
        self.freeze_vision_model = freeze_vision_model
        self.image_pad_num = image_pad_num  # 196 patches / 4 = 49 compressed tokens
        super().__init__(**kwargs)


# ============ Step 1.2: VLM Model ============
class VLM(PreTrainedModel):
    config_class = VLMConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # 视觉编码器 (SigLIP)
        self.vision_model = AutoModel.from_pretrained(config.vision_model_path)
        self.processor = AutoProcessor.from_pretrained(config.vision_model_path)

        # 语言模型 (Qwen2.5-0.5B)
        self.llm_model = AutoModelForCausalLM.from_pretrained(config.llm_model_path, dtype=torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path)

        # 投影层: vision_hidden_size*4 → llm_hidden_size → llm_hidden_size
        self.linear1 = nn.Linear(
            self.vision_model.config.vision_config.hidden_size * 4,
            self.llm_model.config.hidden_size
        )
        self.linear2 = nn.Linear(
            self.llm_model.config.hidden_size,
            self.llm_model.config.hidden_size
        )

        # 冻结策略: 预训练阶段只训练投影层
        if self.config.freeze_vision_model:
            for param in self.vision_model.parameters():
                param.requires_grad = False
        for param in self.llm_model.parameters():
            param.requires_grad = False

    # ============ Step 1.3: Forward ============
    def forward(self, input_ids, labels, pixel_values, attention_mask=None):
        # 文本 → embedding
        text_embeds = self.llm_model.get_input_embeddings()(input_ids)

        # 图片 → 视觉特征 → 压缩patch → 投影
        image_embeds = self.vision_model.vision_model(pixel_values).last_hidden_state  # (b, 196, d)
        image_embeds = rearrange(image_embeds, 'b (s g) d -> b s (g d)', g=4)  # (b, 49, d*4)
        image_features = self.linear2(F.silu(self.linear1(image_embeds)))  # (b, 49, llm_hidden)

        text_embeds = text_embeds.to(image_features.dtype)

        # 图像特征插入文本序列中 <|image_pad|> 的位置
        inputs_embeds = self.merge_input_ids_with_image_features(image_features, text_embeds, input_ids)

        # LLM前向
        outputs = self.llm_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = outputs[0]

        # 计算loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            loss = loss_fct(
                rearrange(logits, 'b s v -> (b s) v'),
                rearrange(labels, 'b s -> (b s)').to(logits.device)
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    # ============ Step 1.4: Merge image features into text embeddings ============
    def merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids):
        # 找到所有 <|image_pad|> 占位符的位置
        batch_indices, image_indices = torch.where(
            input_ids == self.tokenizer('<|image_pad|>')['input_ids'][0]
        )

        # 将图像特征展平后填入对应位置
        inputs_embeds[batch_indices, image_indices] = rearrange(image_features, 'n p d -> (n p) d')

        return inputs_embeds


# ============ Step 1.5: Pre-training Dataset ============
class MyDataset(Dataset):
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
            image_name = sample['image']
            conversations = sample['conversations']

            # 构造问题文本: 套用chat模板，替换<image>为49个<|image_pad|>
            q_text = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": "You are a helpful assistant."},
                 {"role": "user", "content": conversations[0]['value']}],
                tokenize=False,
                add_generation_prompt=True
            ).replace('<image>', '<|image_pad|>' * self.config.image_pad_num)

            # 构造回答文本: 加上eos表示结束
            a_text = conversations[1]['value'] + self.tokenizer.eos_token

            # 分别tokenize后拼接
            q_input_ids = self.tokenizer(q_text)['input_ids']
            a_input_ids = self.tokenizer(a_text)['input_ids']
            input_ids = q_input_ids + a_input_ids

            # labels: 问题部分填pad(不算loss)，回答部分保留
            labels = [self.tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids

            # 错位: 用位置i预测位置i+1
            input_ids = input_ids[:-1]
            labels = labels[1:]

            # 加载图片并预处理
            image = Image.open(os.path.join(self.images_path, image_name)).convert("RGB")
            pixel_values = self.processor(text=None, images=image, return_tensors='pt')['pixel_values']

        except:
            # 图片加载失败时的兜底: 白图+空回答
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


# ============ Step 1.6: Data Collator ============
class MyDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 找到batch内最长的序列长度
        max_len = max(len(feature['input_ids']) for feature in features)
        input_ids = []
        labels = []
        pixel_values = []
        for feature in features:
            # 短的序列用pad_token_id补齐到max_len
            input_ids.append(feature['input_ids'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['input_ids'])))
            labels.append(feature['labels'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['labels'])))
            pixel_values.append(feature['pixel_values'])

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'pixel_values': torch.cat(pixel_values, dim=0)
        }


# ============ Step 1.7: Training Script ============
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_model_path', type=str, required=True)
    parser.add_argument('--vision_model_path', type=str, required=True)
    parser.add_argument('--images_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='save/pretrain')
    args = parser.parse_args()

    # 创建模型
    config = VLMConfig(
        llm_model_path=args.llm_model_path,
        vision_model_path=args.vision_model_path,
        image_pad_num=49
    )
    model = VLM(config).cuda()
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    # 获取tokenizer和processor
    tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path)
    processor = AutoProcessor.from_pretrained(config.vision_model_path)

    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        per_device_train_batch_size=8,
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

    # 创建Trainer并开始训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=MyDataset(args.images_path, args.data_path, tokenizer, processor, config),
        data_collator=MyDataCollator(tokenizer)
    )

    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(args.output_dir)
    trainer.save_state()
