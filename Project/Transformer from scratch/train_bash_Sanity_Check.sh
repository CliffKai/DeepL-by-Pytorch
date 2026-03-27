rm -rf ./checkpoints/sanity_check

uv run python train.py \
    --train_data_path=/root/data/cs336/tinystories_train.npy \
    --val_data_path=/root/data/cs336/tinystories_val.npy \
    --vocab_size=10000 \
    --context_length=256 \
    --d_model=512 \
    --num_layers=4 \
    --num_heads=16 \
    --d_ff=1344 \
    --batch_size=8 \
    --max_steps=100 \
    --eval_interval=10 \
    --log_interval=1 \
    --learning_rate=3e-4 \
    --warmup_steps=0 \

    --out_dir=./checkpoints/sanity_check \
    --device=cuda \
    --wandb_log \
    --wandb_run_name="sanity-check-clean-start"