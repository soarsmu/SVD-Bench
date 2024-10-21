import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--embedding_model', type=str, default='codebert')
parser.add_argument('--num_gpu', type=int, default=8)
parser.add_argument('--few_shot_k', type=int, default=1)
parser.add_argument('--print_only', action='store_true')
parser.add_argument('--down_sampling', action='store_true')
parser.add_argument('--completion_only', action='store_true')
parser.add_argument('--zero_shot', action='store_true')
parser.add_argument('--few_shot_rand', action='store_true')
parser.add_argument('--few_shot_embedding', action='store_true')
parser.add_argument('--fine_tune_slm', action='store_true')
parser.add_argument('--fine_tune_llm_cls', action='store_true')
parser.add_argument('--fine_tune_llm_lm', action='store_true')
args = parser.parse_args()

accelerate_config_path = {
    1: 'configs/accelerate_config_1gpu.yaml',
    2: 'configs/accelerate_config_2gpu.yaml',
    4: 'configs/accelerate_config_4gpu.yaml',
    8: 'configs/accelerate_config_8gpu.yaml',
}[args.num_gpu]

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
        command = (
            f'accelerate launch --config_file {accelerate_config_path} main.py '
            '--eval_only '
            '--task_type generation '
            f'--dataset {args.dataset} '
            f'--model {model} '
        )
        if args.print_only:
            print(command)
        else:
            os.system(command)

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
        command = (
            f'accelerate launch --config_file {accelerate_config_path} main.py '
            '--eval_only '
            f'--few_shot_k {args.few_shot_k} '
            '--max_seq_length 4096 '
            '--per_device_eval_batch_size 4 '
            f'--model {model} '
            f'--dataset {args.dataset} '
        )
        if args.print_only:
            print(command)
        else:
            os.system(command)

############################## few-shot with embedding model ##############################
if args.few_shot_embedding:
    models = [
        'codeqwen1.5-7b-chat',
        'deepseek-coder-6.7b-instruct',
        'codegemma-7b-it',
        'starcoder2-7b',
        'codellama-7b-instruct',
    ]
    for model in models:
        command = (
            f'accelerate launch --config_file {accelerate_config_path} main.py '
            '--eval_only '
            f'--few_shot_k {args.few_shot_k} '
            '--max_seq_length 4096 '
            '--per_device_eval_batch_size 4 '
            f'--embedding_model {args.embedding_model} '
            f'--model {model} '
            f'--dataset {args.dataset} '
        )
        if args.print_only:
            print(command)
        else:
            os.system(command)

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
        command = (
            f'accelerate launch --config_file {accelerate_config_path} main.py '
            '--task_type classification '
            '--max_seq_length 512 '
            '--per_device_train_batch_size 4 '
            f'--dataset {args.dataset} '
            f'--model {model} '
        )
        if args.down_sampling:
            command += ' --down_sampling'
        if args.print_only:
            print(command)
        else:
            os.system(command)

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
        command = (
            f'accelerate launch --config_file {accelerate_config_path} main.py '
            '--task_type classification '
            '--gradient_accumulation_steps 4 ' 
            f'--dataset {args.dataset} '
            f'--model {model} '
        )
        if args.down_sampling:
            command += ' --down_sampling'
        if args.print_only:
            print(command)
        else:
            os.system(command)

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
        command = (
            f'accelerate launch --config_file {accelerate_config_path} main.py '
            '--task_type generation '
            '--gradient_accumulation_steps 4 '
            f'--dataset {args.dataset} '
            f'--model {model} '
        )
        if args.down_sampling:
            command += ' --down_sampling'
        if args.completion_only:
            command += ' --completion_only'
        if args.print_only:
            print(command)
        else:
            os.system(command)
