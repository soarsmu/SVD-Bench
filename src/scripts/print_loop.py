import os
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--few_shot_k", type=int, default=1)
parser.add_argument("--embedding_model", type=str, default='codebert')
parser.add_argument("--result_dir", type=str, default='../results')
parser.add_argument('--down_sampling', action='store_true')
parser.add_argument('--zero_shot', action='store_true')
parser.add_argument('--few_shot_rand', action='store_true')
parser.add_argument('--few_shot_embedding', action='store_true')
parser.add_argument('--fine_tune_slm', action='store_true')
parser.add_argument('--fine_tune_llm_cls', action='store_true')
parser.add_argument('--fine_tune_llm_lm', action='store_true')
args = parser.parse_args()


def print_result(subdir):
    print('-' * 150)
    print(subdir)

    for split in ['valid', 'test']:
        print('-' * 150)
        path = os.path.join(args.result_dir, subdir, f'{split}_results.json')
        try:
            with open(path, 'r') as f:
                metrics = json.load(f)
        except FileNotFoundError:
            print(f'{split} result not found')
            continue

        print(f'{split} accuracy:', metrics[f'{split}_accuracy'])
        print(f'{split} precision:', metrics[f'{split}_precision'])
        print(f'{split} recall:', metrics[f'{split}_recall'])
        print(f'{split} fscore:', metrics[f'{split}_fscore'])

    print('-' * 150)
    print()


############################## zero-shot ##############################
if args.zero_shot:
    models = [
        'codeqwen1.5-7b-chat',
        'deepseek-coder-6.7b-instruct',
        'codegemma-7b-it',
        'starcoder2-7b',
        'codellama-7b-instruct',
    ]
    for model in models:
        for subdir in os.listdir(args.result_dir):
            if '0-shot' not in subdir or 'chat_format' in subdir:
                continue
            if f'{args.dataset}_' not in subdir or f'_{model}_' not in subdir:
                continue

            print_result(subdir)

############################## few-shot with random ##############################
if args.few_shot_rand:
    models = [
        'codeqwen1.5-7b-chat',
        'deepseek-coder-6.7b-instruct',
        'codegemma-7b-it',
        'starcoder2-7b',
        'codellama-7b-instruct',
    ]
    for model in models:
        for subdir in os.listdir(args.result_dir):
            if f'{args.few_shot_k}-shot' not in subdir or 'chat_format' in subdir:
                continue
            if 'simcse' in subdir or 'codebert' in subdir or 'oracle' in subdir:
                continue
            if f'{args.dataset}_' not in subdir or f'_{model}_' not in subdir:
                continue

            print_result(subdir)

############################## few-shot with embedding ##############################
if args.few_shot_embedding:
    models = [
        'codeqwen1.5-7b-chat',
        'deepseek-coder-6.7b-instruct',
        'codegemma-7b-it',
        'starcoder2-7b',
        'codellama-7b-instruct',
    ]
    for model in models:
        for subdir in os.listdir(args.result_dir):
            if f'{args.few_shot_k}-shot' not in subdir or args.embedding_model not in subdir or 'chat_format' in subdir:
                continue
            if f'{args.dataset}_' not in subdir or f'_{model}_' not in subdir:
                continue

            print_result(subdir)

############################## fine-tune with slm ##############################
if args.fine_tune_slm:
    models = [
        'codebert-base',
        'graphcodebert-base',
        'unixcoder-base',
        'codet5-base',
        'codet5p-base',
    ]
    for model in models:
        for subdir in os.listdir(args.result_dir):
            if 'prompt' in subdir:
                continue
            if args.down_sampling and 'balanced' not in subdir:
                continue
            if not args.down_sampling and 'balanced' in subdir:
                continue
            if f'{args.dataset}_' not in subdir or f'_{model}_' not in subdir:
                continue

            print_result(subdir)

############################## fine-tune with llm using seq_cls ##############################
if args.fine_tune_llm_cls:
    models = [
        'codeqwen1.5-7b',
        'deepseek-coder-6.7b-base',
        'codegemma-7b',
        'starcoder2-7b',
        'codellama-7b'
    ]
    for model in models:
        for subdir in os.listdir(args.result_dir):
            if 'prompt' in subdir:
                continue
            if args.down_sampling and 'balanced' not in subdir:
                continue
            if f'{args.dataset}_' not in subdir or f'_{model}_' not in subdir:
                continue

            print_result(subdir)

############################## fine-tune with llm using causal_lm ##############################
if args.fine_tune_llm_lm:
    models = [
        'codeqwen1.5-7b',
        'deepseek-coder-6.7b-base',
        'codegemma-7b',
        'starcoder2-7b',
        'codellama-7b'
    ]
    for model in models:
        for subdir in os.listdir(args.result_dir):
            if 'prompt' not in subdir or 'shot' in subdir:
                continue
            if args.down_sampling and 'balanced' not in subdir:
                continue
            if f'{args.dataset}_' not in subdir or f'_{model}_' not in subdir:
                continue

            print_result(subdir)
