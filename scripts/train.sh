exp_name="qwen2.5_7b_all"
base_model="models/Qwen/Qwen2.5-7B"

# ############################### Train ################################

train_data="dataset/final.jsonl"
output_dir="./checkpoints/$exp_name"

# set large epochs and small batch size for testing
num_train_epochs=1
per_device_train_batch_size=8

num_gpus=8

model_args="\
    --model_name_or_path $base_model \
    --use_lora True \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules q_proj k_proj v_proj o_proj gate_proj down_proj up_proj \
    --additional_special_tokens '<query>' '</query>' '<thought>' '<emb>' \
    --save_merged_lora_model True \
"

data_args="\
    --train_data $train_data \
    --train_group_size 16 \
    --query_max_len 32 \
    --passage_max_len 192 \
    --pad_to_multiple_of 8 \
    --query_instruction_format '<query>{}</query><thought>' \
    --knowledge_distillation True \
    --same_dataset_within_batch True \
    --small_threshold 0 \
    --drop_threshold 0 \
    --example_query_max_len 256
"

training_args="\
    --output_dir $output_dir \
    --overwrite_output_dir \
    --learning_rate 1e-4 \
    --bf16 \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $per_device_train_batch_size \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --deepspeed deepspeed_configs/ds_stage1.json \
    --logging_steps 10 \
    --save_steps 1000 \
    --negatives_cross_device \
    --temperature 0.02 \
    --sentence_pooling_method last_token \
    --normalize_embeddings True \
    --kd_loss_type kl_div \
    --loss_lambda 0.8 \
"

cmd="torchrun --nproc_per_node $num_gpus \
    -m FlagEmbedding.finetune.embedder.decoder_only.o1 \
    $model_args \
    $data_args \
    $training_args \
"

echo $cmd
eval $cmd

# delete checkpoint-*/global_step* dir
rm -rf $output_dir/checkpoint-*/global_step*