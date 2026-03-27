CHECKPOINT_PATH="./checkpoints/lr_sweep_3e-4/ckpt.pt"

uv run python generate.py \
    --checkpoint_path=$CHECKPOINT_PATH \
    --vocab_path=./bpe_tokenizer/tinystories_vocab.json \
    --merges_path=./bpe_tokenizer/tinystories_merges.txt \
    --prompt="Once upon a time, there was a brave knight named" \
    --max_new_tokens=150 \
    --temperature=0.8 \
    --top_p=0.9 \
    --device=cuda