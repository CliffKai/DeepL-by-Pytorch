CHECKPOINT_PATH="./checkpoints/sanity_check/ckpt_step90.pt"

uv run python generate.py \
    --checkpoint_path=$CHECKPOINT_PATH \
    --vocab_path=./bpe_tokenizer/tinystories_vocab.json \
    --merges_path=./bpe_tokenizer/tinystories_merges.txt \
    --prompt="Once upon a time, there was a brave knight named" \
    --max_new_tokens=150 \
    --temperature=0.8 \
    --top_p=0.9 \
    --device=cuda