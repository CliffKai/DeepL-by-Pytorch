# 1.PreTrainedConfig 
`PreTrainedConfig` (定义在 `src/transformers/configuration_utils.py`) 是 Hugging Face `transformers` 库中的一个核心基类。

简单来说，**`PreTrainedConfig` 是模型的“蓝图”或“配方”**。

它是一个 Python 类，唯一的职责就是**存储模型架构的所有超参数**。它不包含任何模型权重或计算逻辑，只包含构建模型所需的所有设置。

  * 当我们创建一个模型时（例如 `BertModel(config)`），模型会查看这个 `config` 对象，并根据其中的属性（如 `config.hidden_size`）来构建其层（Layers）。
  * 当我们保存模型时（`model.save_pretrained("./my-model")`），这个配置对象会被自动保存为一个名为 `config.json` 的文件。
  * 当我们加载模型时（`model = BertModel.from_pretrained("./my-model")`），库会首先加载 `config.json` 文件，将其解析为一个 `BertConfig` 对象，然后使用这个对象作为蓝图来搭建模型的骨架。

### `PreTrainedConfig` 的核心功能

`PreTrainedConfig` 及其子类（如 `BertConfig`, `LlamaConfig` 等）主要负责以下几件事：

1.  **架构定义 (Architecture Definition)**：
    存储构建模型所需的所有参数。这包括：

      * **模型尺寸**: `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `intermediate_size`。
      * **词汇表**: `vocab_size`。
      * **激活函数**: `hidden_act` (例如 "gelu", "silu")。
      * **Dropout 率**: `attention_probs_dropout_prob`, `hidden_dropout_prob`。
      * **特定于模型的参数**: 例如 RoPE 旋转嵌入的 `rope_theta` (Llama) 或 `max_position_embeddings`。

2.  **行为控制 (Behavior Control)**：
    存储控制模型 `forward` 方法输出的标志。

      * `output_attentions` (布尔值): 是否返回所有注意力层的注意力权重。
      * `output_hidden_states` (布尔值): 是否返回所有隐藏层的输出。
      * `return_dict` (布尔值): 是返回一个包含清晰键值对的 `ModelOutput` 对象，还是一个元组 (tuple)。

3.  **任务/微调 (Task/Finetuning)**：
    存储用于特定任务的参数，这些参数通常用于模型的“头部”(Head)。

      * `num_labels`: 分类任务中的标签数量。
      * `id2label` 和 `label2id`: 标签名称和其对应 ID 之间的映射字典。
      * `problem_type`: 用于序列分类，例如 `"single_label_classification"` 或 `"multi_label_classification"`。

4.  **序列化与反序列化 (Serialization)**：
    它提供了将这份“蓝图”保存到磁盘或从磁盘加载的核心方法。

      * `save_pretrained(save_directory)`: 将配置保存为 `save_directory/config.json`。
      * `from_pretrained(path_or_hub_name)`: 从本地路径或 Hub 仓库加载 `config.json`。
      * `to_dict()`: 将配置转换为 Python 字典。
      * `to_json_string()`: 将配置序列化为 JSON 字符串。

5.  **元数据 (Metadata)**：

      * `model_type` (**非常重要**): 这是一个字符串，例如 `"bert"` 或 `"llama"`。这个 `model_type` 是 `AutoConfig` 用来识别应该加载哪个具体配置类（例如 `BertConfig`）的“钥匙”。
      * `architectures`: 一个列表，说明这个配置可以用于哪些模型架构（例如 `["BertForMaskedLM"]`）。
      * `transformers_version`: 保存配置时所用的 `transformers` 库的版本。

### 如何为自定义模型使用 `PreTrainedConfig`？

当我们基于 `transformers` 构建自己的模型时，**必须**为模型创建一个对应的配置类。

假设要创建一个名为 `MyAwesomeTransformer` 的新模型。

#### 第 1 步：创建 `MyAwesomeTransformerConfig`

需要创建一个继承自 `PreTrainedConfig` 的类。

```python
from transformers import PreTrainedConfig

class MyAwesomeTransformerConfig(PreTrainedConfig):
    """
    这是 MyAwesomeTransformer 的配置类。
    它存储了构建 MyAwesomeTransformerModel 所需的所有超参数。
    
    参数:
        vocab_size (int, optional, defaults to 30522):
            词汇表大小。
        hidden_size (int, optional, defaults to 768):
            编码器层和池化层的维度。
        num_hidden_layers (int, optional, defaults to 12):
            Transformer 编码器中的隐藏层数量。
        num_attention_heads (int, optional, defaults to 12):
            Transformer 编码器中每个注意力层的头数。
        my_custom_param (float, optional, defaults to 0.1):
            一个我们自己定义的自定义参数，例如 dropout 率。
        **kwargs:
            传递给基类 PreTrainedConfig 的额外参数。
    """
    
    # model_type 是必须的！这是 AutoConfig 识别模型的关键。
    model_type = "my-awesome-transformer"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        my_custom_param=0.1,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs
    ):
        # 1. 将参数设置到 self
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.my_custom_param = my_custom_param
        
        # 2. 将所有共享的和特殊的参数传递给父类的 __init__
        #    这包括 pad_token_id, eos_token_id, output_attentions, 
        #    num_labels, id2label 等等。
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
```

#### 第 2 步：创建 `MyAwesomeTransformerModel`

在我们的模型类中，将使用这个配置。

```python
from transformers import PreTrainedModel
# 假设在 MyAwesomeTransformerConfig.py 中定义了上面的 Config
# from .configuration_my_awesome_transformer import MyAwesomeTransformerConfig

class MyAwesomeTransformerModel(PreTrainedModel):
    
    # 必须指定配置类
    config_class = MyAwesomeTransformerConfig

    def __init__(self, config: MyAwesomeTransformerConfig):
        # 必须调用 super().__init__(config)
        super().__init__(config)
        
        self.config = config # 存储配置

        # --- 现在，使用 config 属性来构建模型 ---
        # 模型的架构完全由 config 对象驱动
        
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # 举例：一个简单的层
        self.linear_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.my_custom_param)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
        # ... 在这里构建模型的其余部分，
        # 比如一个包含 config.num_hidden_layers 个层的堆栈 ...

        # 确保调用 post_init() 来触发任何最终的权重初始化
        self.post_init()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        # ... 其他输入 ...
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # 0. 检查并使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        
        # 1. 通过嵌入层
        embedding_output = self.embeddings(input_ids)

        # 2. 通过自定义层
        layer_output = self.linear_layer(embedding_output)
        layer_output = self.dropout(layer_output)
        layer_output = self.layer_norm(layer_output)
        
        # ... 模型逻辑 ...
        
        loss = None
        if labels is not None:
            # ... 计算损失 ...
            pass # loss = ...

        # 3. 按要求返回输出
        if not return_dict:
            # 如果 return_dict=False，返回一个元组
            return (layer_output, ...) # (loss, logits, hidden_states, attentions)
            
        # 否则，返回一个 ModelOutput 对象（或字典）
        return {
            "loss": loss,
            "last_hidden_state": layer_output,
            "hidden_states": None, # 如果 output_hidden_states=True，则填充
            "attentions": None,    # 如果 output_attentions=True，则填充
        }

```

#### 第 3 步：（注册到 AutoClass

为了让 `AutoModel.from_pretrained("your-username/my-awesome-transformer")` 能够工作，需要将配置和模型注册到 `AutoConfig` 和 `AutoModel`。

```python
from transformers import AutoConfig, AutoModel

# 将字符串 "my-awesome-transformer" (来自 config.model_type) 
# 映射到配置类
AutoConfig.register("my-awesome-transformer", MyAwesomeTransformerConfig)

# 将配置类映射到模型类
AutoModel.register(MyAwesomeTransformerConfig, MyAwesomeTransformerModel)
```

#### 第 4 步：使用它

现在可以像使用任何其他 Hugging Face 模型一样使用我们自己定义的自定义模型：

```python
# 1. 从头开始创建
my_config = MyAwesomeTransformerConfig(
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    my_custom_param=0.2
)
my_model = MyAwesomeTransformerModel(my_config)

# 2. 保存模型和配置
my_model.save_pretrained("./my-awesome-model-directory")
# 这将在 "./my-awesome-model-directory" 中创建：
# ├── config.json  <-- 由 MyAwesomeTransformerConfig 保存
# └── model.safetensors (或 pytorch_model.bin)

# 3. 重新加载
reloaded_model = MyAwesomeTransformerModel.from_pretrained("./my-awesome-model-directory")

# 4. (如果已注册并上传到 Hub)
# reloaded_model = AutoModel.from_pretrained("your-username/my-awesome-model-directory")
```


# 2.GenerationConfig

### 1. 什么是 GenerationConfig？

简单来说，`GenerationConfig` 是一个**参数“收纳盒”**。

它的唯一作用就是：**存储所有用于控制 `model.generate()` 方法行为的参数**。

当我们调用一个模型（如 Llama, GPT-2, T5）的 `.generate()` 方法来生成文本时，需要指定很多选项，比如：

  * 是用贪心搜索（greedy search）还是采样（sampling）？
  * 如果是采样，`temperature` 是多少？`top_k` 是多少？
  * 如果是束搜索（beam search），`num_beams` 是多少？
  * 生成文本的最大长度（`max_length` 或 `max_new_tokens`）是多少？
  * 用哪个 token 作为 `pad_token_id`、`eos_token_id`？

`GenerationConfig` 类就是把所有这些参数（如 `do_sample`, `num_beams`, `temperature`, `max_length` 等）打包到一个可配置、可保存、可加载的对象中。

#### 关键区别：GenerationConfig vs. PreTrainedConfig

这是一个非常重要的区别：

  * **`PreTrainedConfig`** (来自 `configuration_utils.py`):

      * **定义了模型的“架构”**。
      * 包含像 `hidden_size`, `num_hidden_layers`, `num_attention_heads` 这样的参数。
      * 它决定了模型有多少层、多大，是模型的“蓝图”。你**必须**有这个配置才能加载模型。

  * **`GenerationConfig`** (来自 `generation_configuration_utils.py`):

      * **定义了模型的“生成行为”**。
      * 包含像 `do_sample`, `num_beams`, `temperature` 这样的参数。
      * 它不改变模型的权重或结构，只控制在**推理时**如何从模型中解码出文本。

### 2. GenerationConfig 是如何工作的？

当我们使用 `transformers` 库时，`GenerationConfig` 主要通过以下几种方式与你的模型协同工作：

1.  **自动附加到模型：**

      * 当我们使用 `.from_pretrained()` 加载一个预训练模型时，`transformers` 不仅会下载模型权重（如 `model.safetensors`）和模型配置（`config.json`），它还会自动查找并加载 `generation_config.json` 文件。
      * 加载后，这个配置对象会作为 `model.generation_config` 属性附加到模型实例上。

2.  **作为 `.generate()` 的默认值：**

      * 当我们调用 `model.generate()` 时，该方法会首先查找 `model.generation_config`。
      * `model.generation_config` 中的所有参数都会被用作**默认值**。

3.  **运行时覆盖：**

      * 可以在调用 `.generate()` 时**随时覆盖**这些默认值。
      * 例如，如果 `model.generation_config` 里设置了 `num_beams=1`，但调用时使用了 `model.generate(..., num_beams=5)`，那么这次调用将使用 `num_beams=5`。

### 3. 如何使用它？（4种常见用法）

#### 用法1：隐式使用（最常见）

从 Hub 加载一个模型，它已经附带了一个推荐的 `generation_config.json`。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Mistral-Instruct 的 `generation_config.json` 默认设置了 do_sample=True, top_p=0.95 等
# 当我们调用 .generate() 时，这些设置会自动生效
inputs = tokenizer("Hello, what is", return_tensors="pt")
outputs = model.generate(**inputs) 

print(tokenizer.decode(outputs[0]))
```

  * **发生了什么？** `.generate()` 自动使用了 `model.generation_config` 中为这个指令调优模型预设的采样参数。

#### 用法2：在调用时覆盖默认值

这是最灵活的方式。我们可以使用模型大部分的默认设置，但临时修改一两个参数。

```python
# 接着上面的例子
# 假设我们想强制使用束搜索（beam search）并限制最大长度

outputs_beam = model.generate(
    **inputs, 
    num_beams=4,         # 覆盖默认的 num_beams=1
    do_sample=False,     # 覆盖默认的 do_sample=True
    max_new_tokens=50    # 覆盖默认的 max_length
)

print(tokenizer.decode(outputs_beam[0]))
```

  * **发生了什么？** `.generate()` 使用了 `model.generation_config` 作为基础，然后用传入的 `num_beams=4` 等参数覆盖了它。

#### 用法3：创建并传入一个全新的 GenerationConfig

假设有一套自己非常喜欢的“创意写作”配置，希望在多个模型上复用它。

```python
from transformers import GenerationConfig

# 1. 创建自己的配置对象
creative_config = GenerationConfig(
    do_sample=True,
    temperature=0.9,
    top_k=50,
    top_p=0.9,
    max_new_tokens=256,
    num_return_sequences=3 
)

# 2. 在调用 .generate() 时传入它
# 这将 *完全忽略* model.generation_config，转而使用我们自己定义的 creative_config
creative_outputs = model.generate(
    **inputs,
    generation_config=creative_config
)

for output in creative_outputs:
    print(tokenizer.decode(output))
    print("-" * 20)
```

  * **发生了什么？** 通过 `generation_config=...` 参数 传入了一个完整的配置对象，`.generate()` 会优先使用它。

#### 用法4：保存自己的配置（当我们自己构建模型时）

当我们微调了一个模型，并希望自己或其他人知道如何正确地使用它时，就应该保存一个 `generation_config.json`。

假设我们微调了一个模型用于**摘要**，最好的生成方式是使用束搜索（beam search）。

```python
from transformers import AutoModelForSeq2SeqLM, GenerationConfig

# 1. 加载已经微调好的模型
my_finetuned_model_path = "./my-finetuned-summarizer"
model = AutoModelForSeq2SeqLM.from_pretrained(my_finetuned_model_path)

# 2. 为这个模型定义一个理想的生成配置
# (T5/BART 这类模型通常用束搜索)
config_for_summaries = GenerationConfig(
    num_beams=5,
    max_length=150,
    min_length=40,
    length_penalty=2.0,
    early_stopping=True,
    eos_token_id=model.config.eos_token_id, # 确保设置了正确的EOS token
    pad_token_id=model.config.pad_token_id  # 确保设置了正确的PAD token
)

# 3. 将这个配置保存到模型目录中
config_for_summaries.save_pretrained(
    my_finetuned_model_path
) 
#

# (也可以同时保存模型权重，如果刚训练完)
# model.save_pretrained(my_finetuned_model_path) 
```

  * **发生了什么？**
    1.  创建了一个 `GenerationConfig` 实例，其中包含了推荐的摘要生成参数。
    2.  调用 `.save_pretrained(my_finetuned_model_path)` 会在那个目录中创建一个 `generation_config.json` 文件。
    3.  **未来**：当任何人加载这个模型时（如 `AutoModel.from_pretrained(my_finetuned_model_path)`），他们会自动获得这个配置（如用法1所示），他们的 `.generate()` 调用将自动使用 `num_beams=5` 作为默认值。