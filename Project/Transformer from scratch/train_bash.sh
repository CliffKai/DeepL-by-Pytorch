export WANDB_ENTITY=kkwei0819-beijing-jiaotong-university
export WANDB_PROJECT=cs336_assignment1

LR="3e-4"
# min_lr (通常是 LR / 10)
MIN_LR="3e-5"

echo "============================================================"
echo "Starting training with LR=${LR} and Min_LR=${MIN_LR}"
echo "============================================================"

uv run python train.py \
    --train_data_path=./data/tinystories_train.npy \
    --val_data_path=./data/tinystories_val.npy \
    --vocab_size=10000 \
    --context_length=256 \
    --d_model=512 \
    --num_layers=4 \
    --num_heads=16 \
    --d_ff=1344 \
    --batch_size=64 \
    --max_steps=20000 \
    --warmup_steps=2000 \
    --learning_rate=$LR \
    --min_learning_rate=$MIN_LR \
    --out_dir=./checkpoints/lr_sweep_$LR \
    --device=cuda \
    --wandb_log \
    --wandb_run_name="lr_sweep_$LR"

echo "============================================================"
echo "Training run for LR=${LR} finished."
echo "============================================================"
