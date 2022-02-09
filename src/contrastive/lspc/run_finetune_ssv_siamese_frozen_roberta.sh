#!/bin/bash
#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --export=NONE
BATCH=$1
LR=$2
TEMP=$3
SIZE=$4
AUG=$5
PREAUG=$6
python run_finetune_siamese.py \
	--model_pretrained_checkpoint /your_path/contrastive-product-matching/reports/contrastive/computers-ssv-$SIZE-$PREAUG$BATCH-$LR-$TEMP-roberta-base/pytorch_model.bin \
    --do_train \
    --train_file /your_path/contrastive-product-matching/data/interim/wdc-lspc/training-sets/preprocessed_computers_train_$SIZE.pkl.gz \
	--train_size=$SIZE \
	--validation_file /your_path/contrastive-product-matching/data/interim/wdc-lspc/training-sets/preprocessed_computers_train_$SIZE.pkl.gz \
	--test_file /your_path/contrastive-product-matching/data/interim/wdc-lspc/gold-standards/preprocessed_computers_gs.pkl.gz \
	--evaluation_strategy=epoch \
	--tokenizer="roberta-base" \
	--grad_checkpoint=True \
    --output_dir /your_path/contrastive-product-matching/reports/contrastive-ft-siamese/computers-ssv-$SIZE-$AUG$BATCH-$PREAUG$LR-$TEMP-frozen-roberta-base/ \
	--per_device_train_batch_size=64 \
	--learning_rate=5e-05 \
	--weight_decay=0.01 \
	--num_train_epochs=50 \
	--lr_scheduler_type="linear" \
	--warmup_ratio=0.05 \
	--max_grad_norm=1.0 \
	--fp16 \
	--metric_for_best_model=f1 \
	--dataloader_num_workers=4 \
	--disable_tqdm=True \
	--save_strategy="epoch" \
	--load_best_model_at_end \
	--augment=$AUG \
	#--do_param_opt \