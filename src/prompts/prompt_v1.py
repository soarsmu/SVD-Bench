from functools import partial


categories = ['non-vulnerable', 'vulnerable']
response_template = '\n### Response:\n'


def generate_prompt(sample, is_train, few_shot_samples=None):
    sample['prompt'] = (
        '### Instruction:\n'
        'Analyze the input function, determine if it is vulnerable, '
        'and return the answer as the corresponding label "vulnerable" or "non-vulnerable".'
    )

    if few_shot_samples:
        for few_shot_sample in few_shot_samples:
            label = few_shot_sample['vulnerable']
            input_func = few_shot_sample['function'][:1024]
            sample['prompt'] += (
                '\n### Input:\n'
                '"""\n'
                f'{input_func}\n'
                '"""\n'
                '### Response:\n'
                f'{categories[label]}\n'
            )

    input_func = sample['function'][:1024]
    sample['prompt'] += (
        '\n### Input:\n'
        '"""\n'
        f'{input_func}\n'
        '"""\n'
        '### Response:\n'
    )
    if is_train:
        label = sample['vulnerable']
        sample['prompt'] += categories[label]

    return sample


generate_prompt_train = partial(generate_prompt, is_train=True)
generate_prompt_eval = partial(generate_prompt, is_train=False)
