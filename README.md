# Minimal MPRA-LegNet (LegNet) loader, predictor, and fine-tuner

This repo is a **small, dependency-light** extraction of the pieces you need to:

- **load** a stored MPRA-LegNet / LegNet model (including upstream `human_legnet` Lightning `.ckpt` files)
- **run predictions** for many DNA sequences
- **fine-tune** a loaded model on your own regression dataset using **SGD, Adam, or AdamW** with a **train/val/test split** (val for model selection, test for final evaluation)

It intentionally avoids PyTorch Lightning, pandas, numpy, etc. Only PyTorch is required; Biopython is optionally used for FASTA parsing.

## Quick start

1) Ensure Python 3.9+ and PyTorch are installed. GPU is optional but recommended for speed.
2) From the repo root, explore the CLI help:
   ```bash
   python scripts/predict.py --help
   python scripts/finetune.py --help
   ```
3) (Optional) install in editable mode:
   ```bash
   pip install -e .
   ```

## Concepts: reverse augmentation and reverse channel

- **Reverse-complement augmentation**: train on forward sequences plus their reverse complements.
- **Reverse channel**: optionally add a 5th input channel indicating orientation (all-0 for forward, all-1 for reverse).

This repo preserves these behaviors so you can load upstream checkpoints.

## Predict with a pretrained model

Choose one path based on what you have:

**Path A: You have an upstream `human_legnet` model directory**

Contents expected in `/path/to/model_dir`:

- `config.json`
- one or more `.ckpt` files (often named like `pearson-epoch=..-val_pearson=..ckpt`)

Run predictions:

```bash
python scripts/predict.py \
  --model_dir /path/to/model_dir \
  --input my_sequences.fasta \
  --output preds.tsv \
  --device cuda:0
```

What happens: the script reads `config.json`, auto-picks a checkpoint, runs batched predictions, and (by default) averages forward and reverse-complement predictions if `reverse_augment=true` in the config.

**Path B: You have explicit checkpoint + config paths**

```bash
python scripts/predict.py \
  --checkpoint /path/to/model.ckpt \
  --config /path/to/config.json \
  --input my_sequences.tsv --format table --seq_col sequence \
  --output preds.tsv \
  --device cuda:0
```

**Supported input formats**

- `fasta`: `.fa`, `.fasta`, `.fna`
- `table`: `.tsv` or `.csv` with header row; sequence column defaults to `sequence`
- `text`: any other extension → one sequence per line

## Fine-tune on your regression data

Goal: fine-tune for **200 bp sequences** with **continuous targets**.

1) Prepare data (`.tsv` or `.csv` with header):
   ```text
   sequence\ttarget
   ACGT...\t0.12
   TTGA...\t-1.03
   ...
   ```
2) Run fine-tuning:
   ```bash
   python scripts/finetune.py \
     --model_dir /path/to/pretrained_model_dir \
     --data my_train_data.tsv --seq_col sequence --target_col target \
     --out_dir out_finetune \
     --device cuda:0 \
     --optimizer adamw --lr 1e-4 --weight_decay 1e-3 \
     --epochs 20 --batch_size 256
   ```
3) Inspect outputs in `out_finetune/`:
   - `best_legnet.pt` – self-contained checkpoint (no Lightning needed)
   - `metrics.json` – test MSE and Pearson summary
4) Predict with the fine-tuned checkpoint:
   ```bash
   python scripts/predict.py \
     --checkpoint out_finetune/best_legnet.pt \
     --input my_sequences.fasta \
     --output preds_finetuned.tsv \
     --device cuda:0
   ```

## Tips

- If sequences vary in length, add `--seq_len 200` to pad/truncate.
- `--rc_augment` (train) and `--rc_average` (val/test) default to the upstream config’s `reverse_augment` flag.
- Start with a small subset to verify I/O before long runs.

## License

MIT License (see `LICENSE`). This repo contains an adapted subset of the upstream MPRA-LegNet implementation.
