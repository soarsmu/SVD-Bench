import os
import sys
import time
import random
import logging
import argparse

import numpy as np
from prettytable import PrettyTable
import torch
import datasets
from transformers.utils import logging as transformers_logging
from accelerate.logging import get_logger as accelerate_get_logger


def get_args():
    # set default hyperparameters according to SFTTrainer doc
    # https://huggingface.co/docs/trl/v0.9.6/en/sft_trainer#trl.SFTConfig
    # https://huggingface.co/docs/trl/v0.9.6/en/sft_trainer#trl.ModelConfig
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type', type=str, default='generation',
                        choices=['generation', 'classification'])
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--embedding_model', type=str)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--output_dir', type=str, default='../results',)
    parser.add_argument('--max_seq_length', type=int, default=1024)
    parser.add_argument('--per_device_train_batch_size', type=int, default=4)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--optim', type=str, default='adamw_torch')
    parser.add_argument('--lr_scheduler_type', type=str, default='linear')
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--warmup_ratio', type=float, default=0.0)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--prompt_version', type=int, default=1)
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--few_shot_k', type=int, default=0)
    parser.add_argument('--use_chat_format', default=False, action='store_true')
    parser.add_argument('--down_sampling', default=False, action='store_true')
    parser.add_argument('--dry_run', default=False, action='store_true')
    parser.add_argument('--eval_only', default=False, action='store_true')
    parser.add_argument('--completion_only', default=False, action='store_true')
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(args, is_main_process):
    if not is_main_process:
        transformers_logging.set_verbosity_error()
        transformers_logging.disable_progress_bar()
        datasets.disable_progress_bars()
        os.environ['TQDM_DISABLE'] = '1'
    else:
        transformers_logging.set_verbosity_info()

        console = logging.StreamHandler()
        console.setLevel(level=logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)

        file = logging.FileHandler(os.path.join(args.output_dir, 'logging.log'))
        file.setLevel(level=logging.INFO)
        file.setFormatter(formatter)

        logging.basicConfig(
            level=logging.INFO,
            handlers=[console, file],
        )

    return accelerate_get_logger(__name__)


def get_run_name(args):
    if args.run_name is not None:
        return args.run_name

    if args.checkpoint is not None:
        return args.checkpoint

    if args.task_type == 'generation':
        if args.eval_only:
            tokens = [
                args.dataset,
                args.model,
                f'prompt_v{args.prompt_version}',
                f'{args.few_shot_k}-shot',
                'chat_format' if args.use_chat_format else None,
                args.embedding_model if args.embedding_model else ('rand' if args.few_shot_k else None),
            ]
        else:
            tokens = [
                args.dataset,
                args.model,
                f'prompt_v{args.prompt_version}',
                'balanced' if args.down_sampling else None,
                'completion' if args.completion_only else None,
                args.optim,
                f'lr{args.learning_rate}',
                f'wd{args.weight_decay}',
                f'ep{args.num_train_epochs}',
                f'bs{args.per_device_train_batch_size}',
                f'grad_steps{args.gradient_accumulation_steps}',
                f'lora_r{args.lora_r}',
                f'lora_alpha{args.lora_alpha}'
            ]
    else:
        tokens = [
            args.dataset,
            args.model,
            'balanced' if args.down_sampling else None,
            args.optim,
            f'lr{args.learning_rate}',
            f'wd{args.weight_decay}',
            f'ep{args.num_train_epochs}',
            f'bs{args.per_device_train_batch_size}',
            f'grad_steps{args.gradient_accumulation_steps}',
        ]
    run_name = '_'.join([token for token in tokens if token is not None and token != ''])
    run_name = '{}_{}'.format(run_name, time.strftime('%Y%m%d_%H', time.localtime()))
    return run_name


def log_args(args, logger):
    command = ' '.join(sys.argv)
    logger.info(f'Command: {command}')
    config_table = PrettyTable()
    config_table.field_names = ['Key', 'Value']
    config_table.align['Key'] = 'l'
    config_table.align['Value'] = 'l'
    for config, value in vars(args).items():
        config_table.add_row([config, str(value)])
    logger.info('Configurations:')
    logger.info(config_table)
