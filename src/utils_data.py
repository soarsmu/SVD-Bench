import os
import random
from copy import deepcopy
from pathlib import Path
from collections import defaultdict
import torch
from transformers import DataCollatorForLanguageModeling
from sentence_transformers import SentenceTransformer
from trl import DataCollatorForCompletionOnlyLM
from datasets import Dataset, concatenate_datasets


num_proc = min(os.cpu_count(), 64)


def load_split(dataset_dir, split, remove_columns=False):
    json_path = dataset_dir / f'{split}.json'
    jsonl_path = dataset_dir / f'{split}.jsonl'

    if json_path.is_file():
        path = json_path
    elif jsonl_path.is_file():
        path = jsonl_path
    else:
        raise FileNotFoundError(f'Dataset split not found in either JSON or JSONL format ({json_path}, {jsonl_path})')

    dataset = Dataset.from_json(str(path), split=split)

    if remove_columns:
        dataset = dataset.select_columns(['function', 'vulnerable'])

    return dataset


def get_dataset(dataset_name, dry_run=False, remove_columns=False):
    dataset_dir = Path('../datasets') / dataset_name
    assert dataset_dir.is_dir(), f'Dataset not found ({dataset_dir})'

    dataset_dict = {}
    for split in ['train', 'valid', 'test']:
        dataset_dict[split] = load_split(dataset_dir, split, remove_columns)
        if dry_run:
            num_samples = min(1024, len(dataset_dict[split]))
            dataset_dict[split] = dataset_dict[split].select(range(num_samples))

    return dataset_dict


def down_sample(dataset, target_field='vulnerable', seed=42):
    class_0 = dataset.filter(lambda x: x[target_field] == 0)
    class_1 = dataset.filter(lambda x: x[target_field] == 1)

    min_samples = min(len(class_0), len(class_1))
    downsampled_class_0 = class_0.shuffle(seed=seed).select(range(min_samples))
    downsampled_class_1 = class_1.shuffle(seed=seed).select(range(min_samples))

    balanced_dataset = concatenate_datasets([downsampled_class_0, downsampled_class_1])
    balanced_dataset = balanced_dataset.shuffle(seed=seed)

    return balanced_dataset


def get_data_collator(tokenizer, completion_only, response_template=None):
    if not completion_only:
        return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Using token_ids directly for response_template
    # https://huggingface.co/docs/trl/v0.9.6/en/sft_trainer#using-tokenids-directly-for-responsetemplate
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
    return DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)


def generate_prompt_for_datasets(dataset_dict, prompt_module, few_shot_k=0, embedding_model=None):
    generate_prompt_train = prompt_module.generate_prompt_train
    generate_prompt_eval = prompt_module.generate_prompt_eval

    if few_shot_k:
        # chose examples based on embedding cosine similarity
        if embedding_model and embedding_model != 'oracle':
            embedder_batch_size = 256
            few_shot_candidates = deepcopy(dataset_dict['train'])
            corpus = few_shot_candidates['function']
            embedder = SentenceTransformer(embedding_model)
            corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True, batch_size=embedder_batch_size)
            top_k = min(few_shot_k, len(corpus_embeddings))

            def generate_prompt_retrieve(samples):
                query = samples['function']
                query_embedding = embedder.encode(query, convert_to_tensor=True, show_progress_bar=False)
                similarity_scores = embedder.similarity(query_embedding, corpus_embeddings)
                # Note that for training set, top1 similar example is itself
                scores, indices = torch.topk(similarity_scores, k=top_k)

                samples['prompt'] = []
                for i, index in enumerate(indices):
                    few_shot_samples = few_shot_candidates.select(index)
                    sample = {
                        'function': samples['function'][i],
                        'vulnerable': samples['vulnerable'][i],
                    }
                    sample = generate_prompt(sample, few_shot_samples=few_shot_samples)
                    samples['prompt'].append(sample['prompt'])

                return samples

            # generate zero-shot prompt for train dataset
            dataset_dict['train'] = dataset_dict['train'].map(
                generate_prompt_train, num_proc=num_proc, load_from_cache_file=False
            )
            # generate few-shot prompt for evaluation dataset
            for split in ['valid', 'test']:
                generate_prompt = generate_prompt_train if split == 'train' else generate_prompt_eval
                dataset_dict[split] = dataset_dict[split].map(
                    generate_prompt_retrieve, load_from_cache_file=False, batched=True, batch_size=embedder_batch_size
                )

        # chose examples based on ideal ranking
        elif embedding_model == 'oracle':
            few_shot_candidates = deepcopy(dataset_dict['train'])
            class_0_index = [i for i, sample in enumerate(few_shot_candidates) if sample['vulnerable'] == 0]
            cwe_to_sample = defaultdict(set)
            for i, sample in enumerate(few_shot_candidates):
                if not sample['vulnerable']:
                    continue
                for cwe_id in sample['cwe_id']:
                    cwe_to_sample[cwe_id].add(i)

            def generate_prompt_oracle(sample):
                if not sample['vulnerable']:
                    candidates = class_0_index
                else:
                    assert sample['vulnerable'] == 1
                    candidates = set()
                    for cwe_id in sample['cwe_id']:
                        candidates |= cwe_to_sample[cwe_id]
                    candidates = list(candidates)

                indices = random.sample(candidates, min(len(candidates), few_shot_k))
                # if the number of samples with identical cwe is less than k, append with random examples
                if len(indices) < few_shot_k:
                    others = [i for i in range(len(few_shot_candidates)) if i not in indices]
                    indices += random.sample(others, few_shot_k - len(indices))
                few_shot_samples = few_shot_candidates.select(indices)
                sample = generate_prompt(sample, few_shot_samples=few_shot_samples)
                return sample

            for split in ['train', 'valid', 'test']:
                generate_prompt = generate_prompt_train if split == 'train' else generate_prompt_eval
                dataset_dict[split] = dataset_dict[split].map(
                    generate_prompt_oracle, num_proc=num_proc, load_from_cache_file=False
                )

        # chose examples randomly
        else:
            few_shot_candidates = deepcopy(dataset_dict['train'])

            def generate_prompt_random(sample):
                indices = random.sample(range(len(few_shot_candidates)), few_shot_k)
                few_shot_samples = few_shot_candidates.select(indices)
                sample = generate_prompt(sample, few_shot_samples=few_shot_samples)
                return sample

            for split in ['train', 'valid', 'test']:
                generate_prompt = generate_prompt_train if split == 'train' else generate_prompt_eval
                dataset_dict[split] = dataset_dict[split].map(
                    generate_prompt_random, num_proc=num_proc, load_from_cache_file=False
                )

    else:
        for split in ['train', 'valid', 'test']:
            generate_prompt = generate_prompt_train if split == 'train' else generate_prompt_eval
            # DO NOT LOAD FROM THE CACHE FILE! Otherwise, your updated preprocessing will not take effect
            dataset_dict[split] = dataset_dict[split].map(
                generate_prompt, num_proc=num_proc, load_from_cache_file=False
            )

    return dataset_dict


def tokenize_datasets(dataset_dict, tokenizer, text_field, use_chat_format=False):
    # within T5, it raises ValueError("All examples must have the same number of <eos> tokens.")
    # if there are eos token (i.e. </s>) in the input text, so replace these special token first
    def replace_special_token(examples):
        examples[text_field] = [text.replace(tokenizer.eos_token, '') for text in examples[text_field]]
        return examples

    def apply_chat_format(example):
        tokenized_chat = tokenizer.apply_chat_template(
            example[text_field],
            tokenize=True,
            add_generation_prompt=False,
            return_tensors='pt',
        )
        example[text_field] = tokenizer.decode(tokenized_chat[0])
        return example

    tokenize_batch = lambda examples: (
        tokenizer(examples[text_field], truncation=True, padding=False, max_length=tokenizer.model_max_length)
    )

    for split in ['train', 'valid', 'test']:
        if use_chat_format:
            dataset_dict[split] = dataset_dict[split].map(
                apply_chat_format, num_proc=num_proc, load_from_cache_file=False
            )
        else:
            dataset_dict[split] = dataset_dict[split].map(
                replace_special_token, batched=True, num_proc=num_proc, load_from_cache_file=False
            )
        dataset_dict[split] = dataset_dict[split].map(
            tokenize_batch, batched=True, num_proc=num_proc, load_from_cache_file=False
        )

    return dataset_dict
