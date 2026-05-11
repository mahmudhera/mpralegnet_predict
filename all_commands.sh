# python installations
conda create -n malinois_inference python=3.10 numpy pandas scikit-learn matplotlib tqdm biopython -c conda-forge -c bioconda -y

# load modules and activate conda environment
module use /projects/community/modulefiles
module load git/2.35.1-ez82
module load cuda/12.1.0
module load gcc/10.2.0-bz186
conda activate malinois_inference

# test installation
python scripts/predict_ensemble.py --help

# make inference using a single checkpoint (test run)
python scripts/predict_using_single_model.py --checkpoint /home/mr2320/malinois_inference_minimal/mpralegnet_artifacts/final_dump/models/K562/md_shift_reverse_noavg_noch/best_model_test1_val9.ckpt --config /home/mr2320/malinois_inference_minimal/mpralegnet_artifacts/final_dump/models/K562/md_shift_reverse_noavg_noch/config.json --input /home/mr2320/malinois_inference_minimal/data/sequences_to_predict.txt --no_header --output test.tsv --write_seq

# make inference using an ensemble of checkpoints (test run)
python scripts/predict_ensemble.py --checkpoint_dir /home/mr2320/malinois_inference_minimal/mpralegnet_artifacts/final_dump/models/K562/md_shift_reverse_noavg_noch/ --input /home/mr2320/malinois_inference_minimal/data/sequences_to_predict.txt --no_header --output test.tsv --write_seq

# make variant effect predictions using an ensemble of checkpoints
python scripts/predict_variant_effect_asdvar.py --checkpoint_dir /home/mr2320/malinois_inference_minimal/mpralegnet_artifacts/final_dump/models/K562/md_shift_reverse_noavg_noch/ --input /home/mr2320/malinois_inference_minimal/data/extracted_sequences_variant_analysis.tsv --output results/asdvar_predictions_using_k562.tsv

python scripts/predict_variant_effect_asdvar.py --checkpoint_dir /home/mr2320/malinois_inference_minimal/mpralegnet_artifacts/final_dump/models/HepG2/md_shift_reverse_noavg_noch/ --input /home/mr2320/malinois_inference_minimal/data/extracted_sequences_variant_analysis.tsv --output results/asdvar_predictions_using_hepg2.tsv

python scripts/predict_variant_effect_asdvar.py --checkpoint_dir /home/mr2320/malinois_inference_minimal/mpralegnet_artifacts/final_dump/models/WTC11/md_shift_reverse_noavg_noch/ --input /home/mr2320/malinois_inference_minimal/data/extracted_sequences_variant_analysis.tsv --output results/asdvar_predictions_using_wtc11.tsv

python scripts/predict_variant_effect_mpravardb.py --checkpoint_dir /home/mr2320/malinois_inference_minimal/mpralegnet_artifacts/final_dump/models/K562/md_shift_reverse_noavg_noch/ --input /home/mr2320/malinois_inference_minimal/data/K562_processed.csv --output results/mpravardb_predictions_for_K562_data_using_k562_models.tsv

python scripts/predict_variant_effect_mpravardb.py --checkpoint_dir /home/mr2320/malinois_inference_minimal/mpralegnet_artifacts/final_dump/models/HepG2/md_shift_reverse_noavg_noch/ --input /home/mr2320/malinois_inference_minimal/data/HepG2_processed.csv --output results/mpravardb_predictions_for_HepG2_data_using_hepg2_models.tsv

python scripts/predict_variant_effect_mpac_allelic_skew.py --checkpoint_dir /home/mr2320/malinois_inference_minimal/mpralegnet_artifacts/final_dump/models/K562/md_shift_reverse_noavg_noch/ --input /home/mr2320/malinois_inference_minimal/data/MPAC_emvar_K562_combined.tsv --output results/mpac_allelic_skew_predictions_for_k562_using_k562_models.tsv

python scripts/predict_variant_effect_mpac_allelic_skew.py --checkpoint_dir /home/mr2320/malinois_inference_minimal/mpralegnet_artifacts/final_dump/models/HepG2/md_shift_reverse_noavg_noch/ --input /home/mr2320/malinois_inference_minimal/data/MPAC_emvar_HEPG2_combined.tsv --output results/mpac_allelic_skew_predictions_for_hepg2_using_hepg2_models.tsv



# analysis for these predictions
python scripts/analyze_predictions.py --predictions_file results/asdvar_predictions_using_k562.tsv --ref_logfc_column logFC --pred_logfc_column pred_effect --separator tab
Correlation between logFC and pred_effect: -0.0132
python scripts/analyze_predictions.py --predictions_file results/asdvar_predictions_using_hepg2.tsv --ref_logfc_column logFC --pred_logfc_column pred_effect --separator tab 
Correlation between logFC and pred_effect: -0.0210
python scripts/analyze_predictions.py --predictions_file results/asdvar_predictions_using_wtc11.tsv --ref_logfc_column logFC --pred_logfc_column pred_effect --separator tab 
Correlation between logFC and pred_effect: 0.0047
python scripts/analyze_predictions.py --predictions_file results/mpac_allelic_skew_predictions_for_k562_using_k562_models.tsv --ref_logfc_column log2FC_skew --pred_logfc_column pred_effect --separator tab
Correlation between log2FC_skew and pred_effect: 0.0907
python scripts/analyze_predictions.py --predictions_file results/mpac_allelic_skew_predictions_for_hepg2_using_hepg2_models.tsv --ref_logfc_column log2FC_skew --pred_logfc_column pred_effect --separator tab 
Correlation between log2FC_skew and pred_effect: 0.0718
python scripts/analyze_predictions.py --predictions_file results/mpravardb_predictions_for_K562_data_using_k562_models.tsv --ref_logfc_column log2FC --pred_logfc_column pred_effect --separator tab 
Correlation between log2FC and pred_effect: -0.0144
python scripts/analyze_predictions.py --predictions_file results/mpravardb_predictions_for_HepG2_data_using_hepg2_models.tsv --ref_logfc_column log2FC --pred_logfc_column pred_effect --separator tab 
Correlation between log2FC and pred_effect: 0.0171


# fine tune a model on our MPAC data
python scripts/finetune_for_individual_sequences.py --checkpoint /home/mr2320/malinois_inference_minimal/mpralegnet_artifacts/final_dump/models/K562/md_shift_reverse_noavg_noch/best_model_test1_val9.ckpt --config /home/mr2320/malinois_inference_minimal/mpralegnet_artifacts/final_dump/models/K562/md_shift_reverse_noavg_noch/config.json --data /home/mr2320/malinois_inference_minimal/data/MPAC_emvar_K562_combined.tsv --ref_seq_col ref_seq --ref_activity_col ref_log2FC --alt_seq_col alt_seq --alt_activity_col alt_log2FC --optimizer adamw --lr 0.0001 --batch_size 64 --epochs 50 --select_metric mse --out_model results/finetuned_k562_model.pt --out_metrics results/finetuned_k562_model_metrics.json --seed 777

# retrieve best model and run with smaller learning rate
python scripts/finetune_for_individual_sequences.py --checkpoint results/finetuned_k562_model.pt --config /home/mr2320/malinois_inference_minimal/mpralegnet_artifacts/final_dump/models/K562/md_shift_reverse_noavg_noch/config.json --data /home/mr2320/malinois_inference_minimal/data/MPAC_emvar_K562_combined.tsv --ref_seq_col ref_seq --ref_activity_col ref_log2FC --alt_seq_col alt_seq --alt_activity_col alt_log2FC --optimizer adamw --lr 0.00001 --batch_size 64 --epochs 15 --select_metric mse --out_model results/finetuned_k562_model_v2.pt --out_metrics results/finetuned_k562_model_metrics_v2.json --seed 777

# fine tune a model on ASD_VAR data
python scripts/finetune_for_individual_sequences.py --checkpoint /home/mr2320/malinois_inference_minimal/mpralegnet_artifacts/final_dump/models/K562/md_shift_reverse_noavg_noch/best_model_test1_val9.ckpt --config /home/mr2320/malinois_inference_minimal/mpralegnet_artifacts/final_dump/models/K562/md_shift_reverse_noavg_noch/config.json --data /home/mr2320/malinois_inference_minimal/data/extracted_sequences_variant_analysis.tsv --ref_seq_col ref_sequence --ref_activity_col ref_alpha --alt_seq_col alt_sequence --alt_activity_col alt_alpha --optimizer adamw --lr 0.0001 --batch_size 64 --epochs 50 --select_metric mse --out_model results/finetuned_ASDVAR_model.pt --out_metrics results/finetuned_ASDVAR_model_metrics.json --seed 777

# getting 0.5-ish correlation with alpha values after fine-tuning
# now on to pairwise predictions

python scripts/finetune_for_sequence_pairs.py --checkpoint /home/mr2320/malinois_inference_minimal/mpralegnet_artifacts/final_dump/models/K562/md_shift_reverse_noavg_noch/best_model_test1_val9.ckpt --config /home/mr2320/malinois_inference_minimal/mpralegnet_artifacts/final_dump/models/K562/md_shift_reverse_noavg_noch/config.json --data /home/mr2320/malinois_inference_minimal/data/MPAC_emvar_K562_combined.tsv --ref_seq_col ref_seq --ref_activity_col ref_log2FC --alt_seq_col alt_seq --alt_activity_col alt_log2FC --out_dir results --methods siamese --select_metric mse --rc_pair_augment --rc_average --flip_pairs --epochs 20 --batch_size 256 --optimizer adamw --lr 0.0033 --hidden_dim 128 --dropout 0.1 --loss huber --no-freeze_encoder
# with test: 0.62

python scripts/finetune_for_sequence_pairs.py --checkpoint /home/mr2320/malinois_inference_minimal/mpralegnet_artifacts/final_dump/models/K562/md_shift_reverse_noavg_noch/best_model_test1_val9.ckpt --config /home/mr2320/malinois_inference_minimal/mpralegnet_artifacts/final_dump/models/K562/md_shift_reverse_noavg_noch/config.json --data /home/mr2320/malinois_inference_minimal/data/MPAC_emvar_HEPG2_combined.tsv --ref_seq_col ref_seq --ref_activity_col ref_log2FC --alt_seq_col alt_seq --alt_activity_col alt_log2FC --out_dir results --methods siamese --select_metric mse --rc_pair_augment --rc_average --flip_pairs --epochs 50 --batch_size 256 --optimizer adam --lr 0.0015 --hidden_dim 128 --dropout 0.2 --loss huber --no-freeze_encoder
# with test: 0.62

python scripts/finetune_for_sequence_pairs.py --checkpoint /home/mr2320/malinois_inference_minimal/mpralegnet_artifacts/final_dump/models/K562/md_shift_reverse_noavg_noch/best_model_test1_val9.ckpt --config /home/mr2320/malinois_inference_minimal/mpralegnet_artifacts/final_dump/models/K562/md_shift_reverse_noavg_noch/config.json --data /home/mr2320/malinois_inference_minimal/data/MPAC_emvar_SKNSH_combined.tsv --ref_seq_col ref_seq --ref_activity_col ref_log2FC --alt_seq_col alt_seq --alt_activity_col alt_log2FC --out_dir results --methods siamese --select_metric mse --rc_pair_augment --rc_average --flip_pairs --epochs 20 --batch_size 256 --optimizer adamw --lr 0.0045 --hidden_dim 128 --dropout 0.1 --loss huber --no-freeze_encoder
# with test: 0.59

python scripts/finetune_for_sequence_pairs.py --checkpoint /home/mr2320/malinois_inference_minimal/mpralegnet_artifacts/final_dump/models/K562/md_shift_reverse_noavg_noch/best_model_test1_val9.ckpt --config /home/mr2320/malinois_inference_minimal/mpralegnet_artifacts/final_dump/models/K562/md_shift_reverse_noavg_noch/config.json --data /home/mr2320/malinois_inference_minimal/data/extracted_sequences_variant_analysis.tsv --ref_seq_col ref_sequence --ref_activity_col ref_alpha --alt_seq_col alt_sequence --alt_activity_col alt_alpha --out_dir results --methods siamese --select_metric mse --rc_pair_augment --rc_average --flip_pairs --epochs 20 --batch_size 256 --optimizer adam --lr 0.0015 --hidden_dim 128 --dropout 0.2 --loss huber --no-freeze_encoder --normalize_delta
# 0.12


# now finetune using delta values directly for MPRAVardB data
python scripts/finetune_for_sequence_pairs_using_delta_column.py --checkpoint /home/mr2320/malinois_inference_minimal/mpralegnet_artifacts/final_dump/models/K562/md_shift_reverse_noavg_noch/best_model_test1_val9.ckpt --config /home/mr2320/malinois_inference_minimal/mpralegnet_artifacts/final_dump/models/K562/md_shift_reverse_noavg_noch/config.json --data /home/mr2320/malinois_inference_minimal/data/K562_processed.csv --num_chars_to_ignore 201 --ref_seq_col ref_seq --alt_seq_col alt_seq --delta_col log2FC --out_dir results --methods siamese --select_metric pearson --rc_pair_augment --rc_average --flip_pairs --epochs 30 --batch_size 256 --optimizer adamw --lr 0.0033 --hidden_dim 128 --dropout 0.1 --loss huber --no-freeze_encoder
# 0.44

python scripts/finetune_for_sequence_pairs_using_delta_column.py --checkpoint /home/mr2320/malinois_inference_minimal/mpralegnet_artifacts/final_dump/models/K562/md_shift_reverse_noavg_noch/best_model_test1_val9.ckpt --config /home/mr2320/malinois_inference_minimal/mpralegnet_artifacts/final_dump/models/K562/md_shift_reverse_noavg_noch/config.json --data /home/mr2320/malinois_inference_minimal/data/HepG2_processed.csv --num_chars_to_ignore 201 --ref_seq_col ref_seq --alt_seq_col alt_seq --delta_col log2FC --out_dir results --methods siamese --select_metric pearson --rc_pair_augment --rc_average --flip_pairs --epochs 30 --batch_size 256 --optimizer adamw --lr 0.0033 --hidden_dim 128 --dropout 0.1 --loss huber --no-freeze_encoder
# 0.69


# run for multitask finetuning
python scripts/finetune_for_sequence_pairs_multitask.py --checkpoint /home/mr2320/malinois_inference_minimal/mpralegnet_artifacts/final_dump/models/K562/md_shift_reverse_noavg_noch/best_model_test1_val9.ckpt --config /home/mr2320/malinois_inference_minimal/mpralegnet_artifacts/final_dump/models/K562/md_shift_reverse_noavg_noch/config.json --data /home/mr2320/malinois_inference_minimal/data/MPAC_emvar_HEPG2_combined.tsv --ref_seq_col ref_seq --ref_activity_col ref_log2FC --alt_seq_col alt_seq --alt_activity_col alt_log2FC --out_dir results --select_metric mse --rc_pair_augment --rc_average --flip_pairs --epochs 15 --batch_size 256 --optimizer adam --lr 0.0015 --delta_hidden_dim 128 --delta_dropout 0.2 --loss_ref mse --loss_alt mse --loss_delta mse --no-freeze_encoder


srun --partition=gpu --gres=gpu:4 --pty --mem=48G --exclude=gpu018 -t 120:00 bash
cd mpralegnet_predict
module use /projects/community/modulefiles
module load git/2.35.1-ez82
module load cuda/12.1.0
module load gcc/10.2.0-bz186
conda activate malinois_inference
# train for larger data
python scripts/finetune_for_sequence_pairs.py --checkpoint /home/mr2320/malinois_inference_minimal/mpralegnet_artifacts/final_dump/models/K562/md_shift_reverse_noavg_noch/best_model_test1_val9.ckpt --config /home/mr2320/malinois_inference_minimal/mpralegnet_artifacts/final_dump/models/K562/md_shift_reverse_noavg_noch/config.json --data /scratch/mr2320/process_mpravardb_data/siraj_etal_mpra_data_snv_k562.csv --ref_seq_col ref_200bp --ref_activity_col ref_logFC --alt_seq_col alt_200bp --alt_activity_col alt_logFC --out_dir results --methods siamese --select_metric mse --rc_pair_augment --rc_average --flip_pairs --epochs 20 --batch_size 128 --optimizer adamw --lr 0.001 --hidden_dim 256 --dropout 0.1 --loss huber --no-freeze_encoder


# investigating if the implementations are correct by comparing against Will's predictions
python scripts/predict_ensemble.py --checkpoint_dir /home/mr2320/malinois_inference_minimal/mpralegnet_artifacts/final_dump/models/WTC11/md_shift_reverse_noavg_noch/ --input data/ensg00000253710.fasta --no_header --output data/ensg00000253710_predictions_wtc11.tsv --write_seq --rc_average

python scripts/predict_ensemble.py --checkpoint_dir /home/mr2320/malinois_inference_minimal/mpralegnet_artifacts/final_dump/models/K562/md_shift_reverse_noavg_noch/ --input data/ensg00000253710.fasta --no_header --output data/ensg00000253710_predictions_k562.tsv --write_seq --rc_average

python scripts/predict_ensemble.py --checkpoint_dir /home/mr2320/malinois_inference_minimal/mpralegnet_artifacts/final_dump/models/HepG2/md_shift_reverse_noavg_noch/ --input data/ensg00000253710.fasta --no_header --output data/ensg00000253710_predictions_hepg2.tsv --write_seq --rc_average