"""
Run column type annotation fine-tuning
"""
import numpy as np
np.random.seed(42)
import random
random.seed(42)

import pandas as pd
from sklearn.metrics import classification_report

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import json

from copy import deepcopy

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

from src.contrastive.models.modeling import ContrastiveClassifierModel
from src.contrastive.data.datasets import ContrastiveClassificationDataset
from src.contrastive.data.data_collators import DataCollatorContrastiveClassification
from src.contrastive.models.metrics import compute_metrics_bce

from transformers import EarlyStoppingCallback

from transformers.utils.hp_naming import TrialShortNamer

from pdb import set_trace

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.8.2")

logger = logging.getLogger(__name__)

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
    frozen: Optional[bool] = field(
        default=True, metadata={"help": "If encoder params should be frozen"}
    )
    grad_checkpoint: Optional[bool] = field(
        default=True, metadata={"help": "If aou want to use gradient checkpointing"}
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
    augment: Optional[str] = field(
        default=None, metadata={"help": "The data augmentation to use."}
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

    def get_posneg():
        if data_args.dataset_name == 'amazon-google' or data_args.dataset_name == 'abt-buy':
            return 9
        elif data_args.dataset_name == 'walmart-amazon':
            return 10
        else:
            if data_args.train_size == 'small':
                return 3
            elif data_args.train_size == 'medium':
                return 4
            elif data_args.train_size == 'large':
                return 5
            elif data_args.train_size == 'xlarge':
                return 6

    def model_init(trial):
        init_args = {}
        pos_neg = get_posneg()
        if model_args.model_pretrained_checkpoint:
            my_model = ContrastiveClassifierModel(checkpoint_path=model_args.model_pretrained_checkpoint, len_tokenizer=len(train_dataset.tokenizer), model=model_args.tokenizer, frozen=model_args.frozen, pos_neg=pos_neg, **init_args)
            if model_args.grad_checkpoint:
                my_model.encoder.transformer._set_gradient_checkpointing(my_model.encoder.transformer.encoder, True)
            return my_model
        else:
            my_model = ContrastiveClassifierModel(len_tokenizer=len(train_dataset.tokenizer), model=model_args.tokenizer, frozen=model_args.frozen, pos_neg=pos_neg, **init_args)
            if model_args.grad_checkpoint:
                my_model.encoder.transformer._set_gradient_checkpointing(my_model.encoder.transformer.encoder, True)
            return my_model

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
        train_dataset = ContrastiveClassificationDataset(train_dataset, dataset_type='train', size=data_args.train_size, tokenizer=model_args.tokenizer, dataset=data_args.dataset_name, aug=data_args.augment)
        if training_args.evaluation_strategy != 'no':
            validation_dataset = raw_datasets["validation"]
            validation_dataset = ContrastiveClassificationDataset(validation_dataset, dataset_type='validation', size=data_args.train_size, tokenizer=model_args.tokenizer, dataset=data_args.dataset_name)
        if training_args.load_best_model_at_end:
            test_dataset = raw_datasets["test"]
            test_dataset = ContrastiveClassificationDataset(test_dataset, dataset_type='test', size=data_args.train_size, tokenizer=model_args.tokenizer, dataset=data_args.dataset_name)

    elif training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        validation_dataset = raw_datasets["validation"]
        validation_dataset = ContrastiveClassificationDataset(validation_dataset, dataset_type='validation', size=data_args.train_size, tokenizer=model_args.tokenizer, dataset=data_args.dataset_name)

    elif training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = raw_datasets["test"]
        test_dataset = ContrastiveClassificationDataset(test_dataset, dataset_type='test', size=data_args.train_size, tokenizer=model_args.tokenizer, dataset=data_args.dataset_name)

    # Data collator
    data_collator = DataCollatorContrastiveClassification(tokenizer=train_dataset.tokenizer)

    # Early stopping callback
    callback = EarlyStoppingCallback(early_stopping_patience=10)

    if training_args.do_train and model_args.do_param_opt:

        from ray import tune
        def my_hp_space(trial):
            return {
                "learning_rate": tune.loguniform(5e-5, 5e-3),
                "warmup_ratio": tune.choice([0.05, 0.075, 0.10]),
                "max_grad_norm": tune.choice([0.0, 1.0]),
                "weight_decay": tune.loguniform(0.001, 0.1),
                "seed": tune.randint(1, 50)
            }

        def my_objective(metrics):
            return metrics['eval_f1']

        trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=validation_dataset if training_args.do_eval else None,
        data_collator=data_collator,
        compute_metrics=compute_metrics_bce,
        callbacks=[callback]
        )
        trainer.args.save_total_limit = 1

        def hp_name(trial):
            namer = TrialShortNamer()
            namer.set_defaults('hp', {'learning_rate': 1e-4, 'warmup_ratio': 0.0, 'max_grad_norm': 1.0, 'weight_decay': 0.01, 'seed':1})
            return namer.shortname(trial)

        initial_configs = [
            {
                "learning_rate": 1e-3,
                "warmup_ratio": 0.05,
                "max_grad_norm": 1.0,
                "weight_decay": 0.01,
                "seed": 42
            },
            {
                "learning_rate": 1e-4,
                "warmup_ratio": 0.05,
                "max_grad_norm": 1.0,
                "weight_decay": 0.01,
                "seed": 42
            }
            ]
                
        from ray.tune.suggest.hebo import HEBOSearch
        hebo = HEBOSearch(metric="eval_f1", mode="max", points_to_evaluate=initial_configs, random_state_seed=42)

        best_run = trainer.hyperparameter_search(n_trials=24, direction="maximize", hp_space=my_hp_space, compute_objective=my_objective, backend='ray', 
        resources_per_trial={'cpu':4,'gpu':1}, local_dir=f'{training_args.output_dir}ray_results/', hp_name=hp_name, search_alg=hebo)
        
        with open(f'{training_args.output_dir}best_params.json', 'w') as f:
            json.dump(best_run, f)

    output_dir = deepcopy(training_args.output_dir)
    for run in range(3):
        init_args = {}

        training_args.save_total_limit = 1
        training_args.seed = run
        training_args.output_dir = f'{output_dir}{run}'
        # if model_args.do_param_opt:
        #     init_args = {k:v for k, v in best_run.hyperparameters.items() if k in MODEL_PARAMS}


        # Detecting last checkpoint.
        last_checkpoint = None
        if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(training_args.output_dir)

        pos_neg = get_posneg()
        if model_args.model_pretrained_checkpoint:
            model = ContrastiveClassifierModel(checkpoint_path=model_args.model_pretrained_checkpoint, len_tokenizer=len(train_dataset.tokenizer), model=model_args.tokenizer, frozen=model_args.frozen, pos_neg=pos_neg, **init_args)
            if model_args.grad_checkpoint:
                model.encoder.transformer._set_gradient_checkpointing(model.encoder.transformer.encoder, True)
        else:
            model = ContrastiveClassifierModel(len_tokenizer=len(train_dataset.tokenizer), model=model_args.tokenizer, frozen=model_args.frozen, pos_neg=pos_neg, **init_args)
            if model_args.grad_checkpoint:
                model.encoder.transformer._set_gradient_checkpointing(model.encoder.transformer.encoder, True)

        # Initialize our Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=validation_dataset if training_args.do_eval else None,
            data_collator=data_collator,
            compute_metrics=compute_metrics_bce,
            callbacks=[callback]
        )
        
        # Training
        if training_args.do_train:
            if model_args.do_param_opt:
                for n, v in best_run.hyperparameters.items():
                    setattr(trainer.args, n, v)
                    # if n not in MODEL_PARAMS:
                    #     setattr(trainer.args, n, v)

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

            trainer.log_metrics(f"train", metrics)
            trainer.save_metrics(f"train", metrics)
            trainer.save_state()
        
        # Evaluation
        results = {}
        if training_args.do_eval:
            logger.info("*** Evaluate ***")

            metrics = trainer.evaluate(
                metric_key_prefix="eval"
            )
            max_eval_samples = len(validation_dataset)
            metrics["eval_samples"] = max_eval_samples

            trainer.log_metrics(f"eval", metrics)
            trainer.save_metrics(f"eval", metrics)

        if training_args.do_predict or training_args.do_train:
            logger.info("*** Predict ***")

            predict_results = trainer.predict(
                test_dataset,
                metric_key_prefix="predict"
            )

            metrics = predict_results.metrics
            max_predict_samples = len(test_dataset)
            metrics["predict_samples"] = max_predict_samples

            trainer.log_metrics(f"predict", metrics)
            trainer.save_metrics(f"predict", metrics)

            
    return results

if __name__ == "__main__":
    main()