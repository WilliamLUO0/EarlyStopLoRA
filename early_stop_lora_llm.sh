# Modified by xxxxx on 25/9/2023, based on the script by @artidoro

python es_lora_llm.py \
    --model_name_or_path huggyllama/llama-7b \
    --output_dir ./college_physics/7b/guanaco-7b \
    --logging_steps 10 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 100 \
    --save_total_limit 40 \
    --evaluation_strategy steps \
    --eval_dataset_size 10 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 32 \
    --dataloader_num_workers 3 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --dataset mmlu \
    --target_max_len 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_steps 2000 \
    --eval_steps 20 \
    --learning_rate 4e-6 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.05 \
    --weight_decay 0.0 \
    --seed 0 \
    --do_mmlu_eval \
    --train_subject "college_physics" \
    --early_stop


