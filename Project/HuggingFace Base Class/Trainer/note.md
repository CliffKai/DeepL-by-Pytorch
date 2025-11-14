# 1.`Trainer`

`Trainer` 类是 `transformers` 库中用于模型训练和评估的**核心工具**。

它的目的是将训练过程中的所有样板代码（boilerplate code）抽象出来，让我们不必手动编写训练循环、设备管理、梯度累积、混合精度、分布式训练等复杂逻辑，从而可以**专注于数据处理和模型构建**。

### 1.1 核心功能

`Trainer` 类可以为我们自动处理了以下所有内容：

  * **训练与评估循环**：内置了完整的 `train()`, `evaluate()` 和 `predict()` 循环。
  * **分布式训练**：无缝支持多 GPU 训练 (DataParallel 和 DistributedDataParallel)。
  * **混合精度训练**：通过 `TrainingArguments` 中的 `fp16` 或 `bf16` 标志轻松启用。
  * **梯度累积**：通过 `gradient_accumulation_steps` 参数控制。
  * **优化器与学习率调度器**：默认使用 `AdamW` 优化器和 warmup 调度器，并且可以轻松替换。
  * **日志记录**：自动与 `wandb`, `tensorboard` 等集成，并记录训练损失、评估指标等。
  * **模型保存与断点续训**：在训练过程中按指定策略（如按 `epoch` 或 `steps`）保存 checkpoints，并可以从检查点无缝恢复训练。
  * **与 Accelerate 和 DeepSpeed 集成**：支持高级的分布式训练和显存优化策略。

### 1.2 我们一般如何使用它？

使用 `Trainer` 通常遵循一个标准流程，即“准备组件 -\> 实例化 Trainer -\> 运行”。

#### 1\. 准备组件

在使用 `Trainer` 之前，需要准备好以下几个核心组件：

1.  **Model**：想要训练的 `PreTrainedModel`。
      * 例如: `AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")`
2.  **TrainingArguments**：这是一个**极其重要**的配置类，它包含了训练过程中的**所有**超参数和设置。
      * 例如: `output_dir` (输出目录), `learning_rate` (学习率), `num_train_epochs` (训练轮数), `per_device_train_batch_size` (批次大小), `evaluation_strategy` (评估策略，如 "steps" 或 "epoch"), `save_strategy` (保存策略)等。
3.  **Datasets**：`train_dataset` 和 `eval_dataset`。
4.  **Processing Class**：通常是 `Tokenizer` 或 `Processor`，用于数据预处理。`Trainer` 会使用它来确保模型和数据在保存和加载时的一致性。
5.  **DataCollator**：一个函数，用于将数据集中的样本列表打包成一个批次（batch）。
      * 例如: `DataCollatorWithPadding` 会自动将批次内的样本填充到最大长度。
6.  **(可选) compute\_metrics (评估函数)**：我们自己自定义的一个函数，它接收一个 `EvalPrediction` 对象（包含 `predictions` 和 `label_ids`），并返回一个包含评估指标（如 "accuracy", "f1"）的字典。

#### 2\. 实例化 Trainer

准备好所有组件后，只需将它们传递给 `Trainer` 的构造函数 `__init__` 即可。

```python
from transformers import Trainer, TrainingArguments

# 1. 假设 model, train_dataset, eval_dataset, tokenizer, data_collator, compute_metrics 都已定义

# 2. 定义训练参数
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",  # 每个 epoch 结束后进行评估
    logging_steps=100,
    save_strategy="epoch",        # 每个 epoch 结束后保存模型
    load_best_model_at_end=True,  # 训练结束后加载最佳模型
)

# 3. 实例化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=tokenizer,  # 在 v4.49+ 中，建议使用 processing_class
)
```

#### 3\. 运行训练和评估

实例化之后，只需调用相应的方法：

  * **`trainer.train()`**：
    启动完整的训练过程。`Trainer` 会自动处理所有训练循环、梯度计算、优化器步骤、学习率调度、日志记录和模型保存。
  * **`trainer.evaluate()`**：
    在 `eval_dataset` 上运行评估。
  * **`trainer.predict(test_dataset)`**：
    在 `test_dataset` 上生成预测结果。

