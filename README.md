# Minimal MPRA-LegNet (LegNet) loader, predictor, and fine-tuner

This repo is a **small, dependency-light** extraction of the pieces you need to:

- **load** a stored MPRA-LegNet / LegNet model (including upstream `human_legnet` Lightning `.ckpt` files)
- **run predictions** for many DNA sequences
- **fine-tune** a loaded model on your own regression dataset using **SGD, Adam, or AdamW**
  with a **train/val/test split** (val for model selection, test for final evaluation).

It intentionally avoids PyTorch Lightning, pandas, numpy, etc. (Only PyTorch is required; Biopython is *optionally* used for FASTA parsing.)

## What “reverse augmentation” / “reverse channel” means

Upstream MPRA-LegNet uses two related options:

- **Reverse-complement augmentation**: train on forward sequences plus their reverse complements.
- **Reverse channel**: optionally add a **5th input channel** indicating orientation (all-0 for forward, all-1 for reverse).

This repo preserves these behaviors so you can load upstream checkpoints.

## Install (optional)

You can run the scripts without installation. From the repo root:

```bash
python scripts/predict.py --help
python scripts/finetune.py --help
```

If you prefer installation:

```bash
pip install -e .
```

## Predict with a pretrained model

### A) If you have an upstream `human_legnet` model directory

Upstream training produces a directory with:

- `config.json`
- one or more `.ckpt` files (often named like `pearson-epoch=..-val_pearson=..ckpt`)

Run:

```bash
python scripts/predict.py \
  --model_dir /path/to/model_dir \
  --input my_sequences.fasta \
  --output preds.tsv \
  --device cuda:0
```

The script will:

- read `/path/to/model_dir/config.json`
- auto-pick a checkpoint under that directory
- run batched GPU predictions
- (by default) average forward and reverse-complement predictions if `reverse_augment=true` in the config.

### B) If you have explicit checkpoint + config paths

```bash
python scripts/predict.py \
  --checkpoint /path/to/model.ckpt \
  --config /path/to/config.json \
  --input my_sequences.tsv --format table --seq_col sequence \
  --output preds.tsv \
  --device cuda:0
```

### Supported input formats

- **FASTA**: `.fa`, `.fasta`, `.fna`
- **Table**: `.tsv` or `.csv` with a header row and a sequence column (default name: `sequence`)
- **Text**: any other extension → one sequence per line

## Fine-tune on your regression data

Your dataset: **200 bp sequences** with **continuous targets**.

Prepare a TSV (or CSV) with header, e.g.:

```text
sequence\ttarget
ACGT...\t0.12
TTGA...\t-1.03
...
```

Run fine-tuning:

```bash
python scripts/finetune.py \
  --model_dir /path/to/pretrained_model_dir \
  --data my_train_data.tsv --seq_col sequence --target_col target \
  --out_dir out_finetune \
  --device cuda:0 \
  --optimizer adamw --lr 1e-4 --weight_decay 1e-3 \
  --epochs 20 --batch_size 256
```

Outputs:

- `out_finetune/best_legnet.pt` – a **self-contained** checkpoint (no Lightning needed to reload)
- `out_finetune/metrics.json` – a small summary with test MSE and Pearson

To predict with the fine-tuned checkpoint:

```bash
python scripts/predict.py \
  --checkpoint out_finetune/best_legnet.pt \
  --input my_sequences.fasta \
  --output preds_finetuned.tsv \
  --device cuda:0
```

## Notes

- If your sequences are not all the same length, use `--seq_len 200` to pad/truncate.
- By default, `--rc_augment` (training) and `--rc_average` (val/test) follow the upstream config’s `reverse_augment` flag.

## License

MIT License (see `LICENSE`). This repo contains an adapted subset of the upstream MPRA-LegNet implementation.
