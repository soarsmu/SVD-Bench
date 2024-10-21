## How to Use

### Fine-tune using SLM with classification loss
```
accelerate launch \
    --config_file configs/accelerate_config_8gpu.yaml \
    main.py \
    --task_type classfication \
    --dataset javascript \
    --model codebert-base \
    --max_seq_length 512
```

### Zero-shot using LLM
```
accelerate launch \
    --config_file configs/accelerate_config_8gpu.yaml \
    main.py \
    --task_type generation \
    --dataset javascript \
    --model codeqwen1.5-7b-chat \
    --prompt_version 1 \
    --eval_only
```

### Few-shot using LLM with random examples
```
accelerate launch \
    --config_file configs/accelerate_config_8gpu.yaml \
    main.py \
    --task_type generation \
    --dataset javascript \
    --model codeqwen1.5-7b-chat \
    --prompt_version 1 \
    --eval_only \
    --few_shot_k 3 \
    --max_seq_length 4096 \ 
    --per_device_eval_batch_size 4
```

### Few-shot using LLM with similar examples based on embedding model
```
accelerate launch \
    --config_file configs/accelerate_config_8gpu.yaml \
    main.py \
    --task_type generation \
    --dataset javascript \
    --model codeqwen1.5-7b-chat \
    --prompt_version 1 \
    --eval_only \
    --few_shot_k 3 \
    --embedding_model simcse-bert-base
```

### Fine-tune (QLoRA) using LLM with causal LM loss
```
accelerate launch \
    --config_file configs/accelerate_config_8gpu.yaml \
    main.py \
    --task_type generation \
    --dataset javascript \
    --model codeqwen1.5-7b \
    --prompt_version 1
```

### Fine-tune (QLoRA) using LLM with classification loss
```
accelerate launch \
    --config_file configs/accelerate_config_8gpu.yaml \
    main.py \
    --task_type classfication \
    --dataset javascript \
    --model codeqwen1.5-7b
```
