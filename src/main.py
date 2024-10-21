import os
import importlib
from functools import partial

from trl import SFTConfig
from transformers import DataCollatorWithPadding, Trainer
from accelerate import Accelerator

from utils import (
    get_args,
    set_seed,
    get_logger,
    get_run_name,
    log_args
)
from utils_data import (
    num_proc,
    get_dataset,
    get_data_collator,
    down_sample,
    generate_prompt_for_datasets,
    tokenize_datasets
)
from utils_model import (
    model_dict_slm,
    model_dict_llm,
    model_dict_embedding,
    load_tokenizer_and_model_slm,
    load_tokenizer_and_model_llm_causal_lm,
    load_tokenizer_and_model_llm_seq_cls,
    CustomSFTTrainer,
    compute_metrics_lm,
    compute_metrics_cls,
)


if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)
    accelerator = Accelerator()

    args.run_name = get_run_name(args)
    if '../results' in args.run_name:
        args.output_dir = args.run_name
    else:
        args.output_dir = os.path.join(args.output_dir, args.run_name)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(args, accelerator.is_main_process)
    log_args(args, logger)

    logger.info('=' * 20 + ' Loading dataset ' + '=' * 20)
    dataset_dict = get_dataset(args.dataset, args.dry_run)
    if args.down_sampling:
        dataset_dict['train'] = down_sample(dataset_dict['train'])
    logger.info(f'Train set size: {len(dataset_dict["train"])}')
    logger.info(f'Valid set size: {len(dataset_dict["valid"])}')
    logger.info(f' Test set size: {len(dataset_dict["test"])}')

    logger.info('=' * 20 + ' Loading tokenizer and model ' + '=' * 20)
    if args.task_type == 'generation':
        tokenizer, model = load_tokenizer_and_model_llm_causal_lm(args)
        if accelerator.is_main_process and not args.eval_only:
            model.print_trainable_parameters()
    else:
        assert args.task_type == 'classification'
        if args.model in model_dict_llm:
            tokenizer, model = load_tokenizer_and_model_llm_seq_cls(args)
            if accelerator.is_main_process and not args.eval_only:
                model.print_trainable_parameters()
        else:
            assert args.model in model_dict_slm
            tokenizer, model = load_tokenizer_and_model_slm(args)

    logger.info('=' * 20 + ' Preprocessing dataset ' + '=' * 20)
    if args.task_type == 'generation':
        prompt_modules = [f'prompts.prompt_v{i}' for i in [0, 1, 2, 3]]
        prompt_module = importlib.import_module(prompt_modules[args.prompt_version])
        response_template = prompt_module.response_template

        args.embedding_model = model_dict_embedding.get(args.embedding_model, args.embedding_model)
        dataset_dict = generate_prompt_for_datasets(dataset_dict, prompt_module, args.few_shot_k, args.embedding_model)
        dataset_dict = tokenize_datasets(dataset_dict, tokenizer, text_field='prompt', use_chat_format=args.use_chat_format)
        data_collator = get_data_collator(tokenizer, args.completion_only, response_template)

        # reserve original binary labels to compute metrics
        y_true_valid = dataset_dict['valid']['vulnerable']
        y_true_test = dataset_dict['test']['vulnerable']
        compute_metrics_valid = partial(
            compute_metrics_lm, tokenizer=tokenizer, response_template=response_template, y_true=y_true_valid,
            output_dir=f'{args.output_dir}/pred_valid', beta=args.beta
        )
        compute_metrics_test = partial(
            compute_metrics_lm, tokenizer=tokenizer, response_template=response_template, y_true=y_true_test,
            output_dir=f'{args.output_dir}/pred_test', beta=args.beta
        )
    else:
        assert args.task_type == 'classification'
        dataset_dict = tokenize_datasets(dataset_dict, tokenizer, text_field='function')
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        for split in ['train', 'valid', 'test']:
            dataset_dict[split] = dataset_dict[split].rename_column('vulnerable', 'label')
        compute_metrics_valid = partial(compute_metrics_cls, output_dir=f'{args.output_dir}/pred_valid', beta=args.beta)
        compute_metrics_test = partial(compute_metrics_cls, output_dir=f'{args.output_dir}/pred_test', beta=args.beta)

    logger.info('=' * 20 + ' Initialize trainer ' + '=' * 20)
    training_args = SFTConfig(
        output_dir=args.output_dir,
        report_to='all',
        dataset_text_field='prompt',
        dataloader_num_workers=num_proc,
        max_seq_length=tokenizer.model_max_length,
        save_strategy='epoch',
        eval_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='fscore',
        greater_is_better=True,
        log_on_each_node=False,
        fp16=False,
        bf16=True,
        ddp_find_unused_parameters=False,
        skip_memory_metrics=True,
        logging_steps=25,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optim,
        lr_scheduler_type=args.lr_scheduler_type,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
    )
    trainer = (CustomSFTTrainer if args.task_type == 'generation' else Trainer)(
        model,
        args=training_args,
        train_dataset=dataset_dict['train'],
        eval_dataset=dataset_dict['valid'],
        data_collator=data_collator,
        compute_metrics=compute_metrics_valid,
    )

    if not args.eval_only:
        logger.info('=' * 20 + ' Training ' + '=' * 20)
        trainer.train()

    logger.info('=' * 20 + ' Evaluating on validation set ' + '=' * 20)
    metrics = trainer.evaluate(dataset_dict['valid'], metric_key_prefix='valid')
    trainer.log_metrics(split='valid', metrics=metrics)
    trainer.save_metrics(split='valid', metrics=metrics)

    logger.info('=' * 20 + ' Evaluating on test set ' + '=' * 20)
    trainer.compute_metrics = compute_metrics_test
    metrics = trainer.evaluate(dataset_dict['test'], metric_key_prefix='test')
    trainer.log_metrics(split='test', metrics=metrics)
    trainer.save_metrics(split='test', metrics=metrics)
