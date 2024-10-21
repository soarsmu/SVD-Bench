import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSequenceClassification
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model
from accelerate import PartialState
from trl import SFTTrainer, setup_chat_format


model_dict_llm = {
    'codeqwen1.5-7b': ('Qwen/CodeQwen1.5-7B', 65536),
    'codeqwen1.5-7b-chat': ('Qwen/CodeQwen1.5-7B-Chat', 65536),
    'deepseek-coder-6.7b-base': ('deepseek-ai/deepseek-coder-6.7b-base', 16384),
    'deepseek-coder-6.7b-instruct': ('deepseek-ai/deepseek-coder-6.7b-instruct', 16384),
    'codegemma-7b': ('google/codegemma-7b', 8192),
    'codegemma-7b-it': ('google/codegemma-7b-it', 8192),
    'starcoder2-7b': ('bigcode/starcoder2-7b', 16384),
    'codellama-7b': ('codellama/CodeLlama-7b-hf', 16384),
    'codellama-7b-instruct': ('codellama/CodeLlama-7b-Instruct-hf', 16384),
}

model_dict_slm = {
    'codebert-base': 'microsoft/codebert-base',
    'graphcodebert-base': 'microsoft/graphcodebert-base',
    'unixcoder-base': 'microsoft/unixcoder-base',
    'codet5-base': 'Salesforce/codet5-base',
    'codet5p-base': 'Salesforce/codet5p-220m',
}

model_dict_embedding = {
    'simcse-bert-base': 'princeton-nlp/sup-simcse-bert-base-uncased',
    'simcse-roberta-base': 'princeton-nlp/sup-simcse-roberta-base',
    'codebert-base': 'microsoft/codebert-base',
    'jina-v2-base-code': 'jinaai/jina-embeddings-v2-base-code',
}


NUM_LABELS = 2
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


def load_tokenizer_and_model_slm(args):
    assert args.model in model_dict_slm
    model_name = model_dict_slm[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.model_max_length = min(tokenizer.model_max_length, args.max_seq_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        trust_remote_code=True
    )
    return tokenizer, model


def load_tokenizer_llm(args):
    assert args.model in model_dict_llm
    model_name, model_max_length = model_dict_llm[args.model]
    # use small max_seq_len to avoid GPU OOM
    model_max_length = min(model_max_length, args.max_seq_length)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint or model_name, trust_remote_code=True)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.model_max_length = model_max_length
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    return tokenizer


def setup_peft_model(args, model):
    task_type = {
        'classification': 'SEQ_CLS',
        'generation': 'CAUSAL_LM',
    }[args.task_type]

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias='none',
        task_type=task_type,
        target_modules='all-linear',
    )
    model = get_peft_model(model, lora_config)

    return model


def load_tokenizer_and_model_llm_seq_cls(args):
    tokenizer = load_tokenizer_llm(args)

    # Ensure that the model is placed on the correct device for multi-GPU training
    # https://huggingface.co/docs/trl/v0.9.6/en/sft_trainer#multi-gpu-training
    device_string = PartialState().process_index
    device_map = {'': device_string}

    if args.checkpoint:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.checkpoint,
            device_map=device_map,
        )
        return tokenizer, model

    assert args.model in model_dict_llm
    model_name, _ = model_dict_llm[args.model]
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        torch_dtype='auto',
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=bnb_config,
    )
    model.config.pad_token_id = model.config.eos_token_id
    model = setup_peft_model(args, model)

    return tokenizer, model


def load_tokenizer_and_model_llm_causal_lm(args):
    tokenizer = load_tokenizer_llm(args)

    # Ensure that the model is placed on the correct device for multi-GPU training
    # https://huggingface.co/docs/trl/v0.9.6/en/sft_trainer#multi-gpu-training
    device_string = PartialState().process_index
    device_map = {'': device_string}

    if args.checkpoint:
        model = AutoPeftModelForCausalLM.from_pretrained(
            args.checkpoint,
            device_map=device_map,
        )
        return tokenizer, model

    assert args.model in model_dict_llm
    model_name, _ = model_dict_llm[args.model]
    if args.eval_only:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype='auto',
            device_map=device_map,
            trust_remote_code=True,
        )
        return tokenizer, model

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype='auto',
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=bnb_config,
    )

    if args.use_chat_format:
        model, tokenizer = setup_chat_format(model, tokenizer)
    model = setup_peft_model(args, model)

    return tokenizer, model


class CustomSFTTrainer(SFTTrainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Return: A tuple with the loss, logits and labels.
        """
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=False,
                top_p=None,  # set top_p to None to avoid warning
                max_new_tokens=5,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

        loss = None
        # to compute metrics, labels should not be None, set as placeholders instead
        labels = [-1 for _ in range(len(inputs['input_ids']))]
        labels = torch.tensor(labels).to(model.device)

        return loss, outputs, labels


def compute_metrics_lm(eval_prediction, response_template, tokenizer, y_true, output_dir=None, beta=.3):
    outputs = eval_prediction.predictions
    # Trainer pads outputs with -100, replace them with pad_token first
    # https://github.com/huggingface/transformers/issues/22634
    outputs = np.where(outputs != -100, outputs, tokenizer.pad_token_id)
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    responses = []
    for output in outputs:
        if response_template in output:
            response = output.split(response_template)[-1].strip()
        else:
            # the response part can be truncated if the input function is too long, but this is rare
            response = output
        responses.append(response)

    # convert response to binary label
    y_pred = []
    for response in responses:
        if 'non' in response or 'false' in response or '0' in response or 'no' in response or 'NO' in response:
            y_pred.append(0)
        elif 'vul' in response or 'true' in response or '1' in response or 'yes' in response or 'YES' in response:
            y_pred.append(1)
        else:
            # -1 as null label
            y_pred.append(-1)

    y_pred = np.array(y_pred)
    null_pred_ratio = np.mean(y_pred == -1)
    # treat null label as negative label (non-vulnerable)
    y_pred = np.where(y_pred == -1, 0, y_pred)
    y_pred = y_pred.tolist()
    assert len(y_pred) == len(y_true)

    accuracy = accuracy_score(y_true, y_pred)
    # compute precision, recall and fscore
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', beta=beta)
    metrics = {
        'null_pred_ratio': null_pred_ratio,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'fscore': fscore,
    }

    # save labels, outputs, responses and predictions
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        metrics.update(y_true=y_true, y_pred=y_pred, responses=responses, outputs=outputs)
        for field in ['y_true', 'y_pred', 'responses', 'outputs']:
            with open(f'{output_dir}/{field}.json', 'w') as f:
                json.dump(metrics.pop(field), f, indent=4)

    return metrics


def compute_metrics_cls(eval_prediction, output_dir=None, beta=.3):
    y_true = eval_prediction.label_ids
    logits = eval_prediction.predictions
    # T5 return tuple as result
    if isinstance(logits, tuple):
        logits = logits[0]
    y_pred = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(y_true, y_pred)
    # compute precision, recall and fscore
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', beta=beta)
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'fscore': fscore,
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(f'{output_dir}/y_true.json', 'w') as f:
            json.dump(y_true.tolist(), f, indent=4)
        with open(f'{output_dir}/y_pred.json', 'w') as f:
            json.dump(y_pred.tolist(), f, indent=4)

    return metrics
