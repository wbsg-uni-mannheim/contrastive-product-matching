"""
Run contrastive pre-training
"""
import numpy as np
np.random.seed(42)
import random
random.seed(42)

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import json

import torch

import transformers as transformers

from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed
)
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from src.contrastive.models.modeling import ContrastivePretrainModel
from src.contrastive.data.datasets import ContrastivePretrainDatasetDeepmatcher
from src.contrastive.data.data_collators import DataCollatorContrastivePretrainDeepmatcher
from src.contrastive.models.metrics import compute_metrics_bce

from transformers import EarlyStoppingCallback

from pdb import set_trace

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.8.2")

logger = logging.getLogger(__name__)

MODEL_PARAMS=['pool']

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_pretrained_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    do_param_opt: Optional[bool] = field(
        default=False, metadata={"help": "If aou want to do hyperparamter optimization"}
    )
    grad_checkpoint: Optional[bool] = field(
        default=True, metadata={"help": "If aou want to use gradient checkpointing"}
    )
    temperature: Optional[float] = field(
        default=0.07,
        metadata={
            "help": "Temperature for contrastive loss"
        },
    )
    tokenizer: Optional[str] = field(
        default='huawei-noah/TinyBERT_General_4L_312D',
        metadata={
            "help": "Tokenizer to use"
        },
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    interm_file: Optional[str] = field(
        default=None, metadata={"help": "The intermediate training set."}
    )
    clean: Optional[bool] = field(
        default=False, metadata={"help": "Only use intermediate training set"}
    )
    augment: Optional[str] = field(
        default=None, metadata={"help": "The data augmentation to use."}
    )
    id_deduction_set: Optional[str] = field(
        default=None, metadata={"help": "The size of the training set."}
    )
    train_size: Optional[str] = field(
        default=None, metadata={"help": "The size of the training set."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
            "(a jsonlines or csv file)."
        },
    )
    max_validation_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    dataset_name: Optional[str] = field(
        default='lspc',
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
            "(a jsonlines or csv file)."
        },
    )
    def __post_init__(self):
        if self.train_file is None and self.validation_file is None:
            raise ValueError("Need a training file.")



def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
    raw_datasets = data_files

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.interm_file is not None:
            train_dataset = ContrastivePretrainDatasetDeepmatcher(train_dataset, tokenizer=model_args.tokenizer, intermediate_set=data_args.interm_file, clean=data_args.clean, dataset=data_args.dataset_name, deduction_set=data_args.id_deduction_set, aug=data_args.augment, split=False)
        else:
            train_dataset = ContrastivePretrainDatasetDeepmatcher(train_dataset, tokenizer=model_args.tokenizer, clean=data_args.clean, dataset=data_args.dataset_name, deduction_set=data_args.id_deduction_set, aug=data_args.augment, split=False)

    # Data collator
    data_collator = DataCollatorContrastivePretrainDeepmatcher(tokenizer=train_dataset.tokenizer)

    if model_args.model_pretrained_checkpoint:
        model = ContrastivePretrainModel(model_args.model_pretrained_checkpoint, len_tokenizer=len(train_dataset.tokenizer), model=model_args.tokenizer, temperature=model_args.temperature)
        if model_args.grad_checkpoint:
            model.encoder.transformer._set_gradient_checkpointing(model.encoder.transformer.encoder, True)
    else:
        model = ContrastivePretrainModel(len_tokenizer=len(train_dataset.tokenizer), model=model_args.tokenizer, temperature=model_args.temperature)
        if model_args.grad_checkpoint:
            model.encoder.transformer._set_gradient_checkpointing(model.encoder.transformer.encoder, True)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=validation_dataset if training_args.do_eval else None,
        data_collator=data_collator,
        compute_metrics=compute_metrics_bce
    )
    trainer.args.save_total_limit = 1

    # Training
    if training_args.do_train:

        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

if __name__ == "__main__":
    main()