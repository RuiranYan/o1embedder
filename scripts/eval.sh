exp_name="qwen2.5_7b_all"
base_model="models/Qwen/Qwen2.5-7B"

# ############################### Eval BEIR ################################

# dataset_names="fiqa nfcorpus scidocs scifact trec-covid webis-touche2020 dbpedia-entity nq hotpotqa fever"
dataset_names="fiqa"
# model_name_or_path="./checkpoints/$exp_name/merged_model"
model_name_or_path="Ruiran/o1embedder"

corpus_embd_save_dir="./beir/$exp_name/corpus_embd"
output_dir="./beir/$exp_name/search_results"
eval_output_path="./beir/$exp_name/beir_eval_results.md"


if [ -z "$HF_HUB_CACHE" ]; then
    # export HF_HUB_CACHE="~/.cache/huggingface/hub"
    export HF_HUB_CACHE="/share/shitao/wyz/yrr/models/huggingface/hub"
fi


eval_args="\
    --eval_name beir \
    --dataset_dir ./beir/data \
    --dataset_names $dataset_names \
    --splits test dev \
    --corpus_embd_save_dir ./beir/$exp_name/corpus_embd \
    --output_dir $output_dir \
    --search_top_k 1000 \
    --cache_path $HF_HUB_CACHE \
    --overwrite True \
    --k_values 10 100 1000 \
    --eval_output_method markdown \
    --eval_output_path $eval_output_path \
    --eval_metrics ndcg_at_10 recall_at_1000 mrr_at_10\
    --ignore_identical_ids True \
"

model_args="\
    --embedder_name_or_path $model_name_or_path \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
    --cache_dir $HF_HUB_CACHE \
    --embedder_query_max_length 1024 \
    --embedder_batch_size 128 \
"

cmd="python -m FlagEmbedding.evaluation.beir \
    $eval_args \
    $model_args \
"

echo $cmd
eval $cmd


############################## Eval MSMARCO ################################

dataset_names="passage"

model_name_or_path="./checkpoints/$exp_name/merged_model"

corpus_embd_save_dir="./msmarco/$exp_name/corpus_embd"
output_dir="./msmarco/$exp_name/search_results"
eval_output_path="./msmarco/$exp_name/msmarco_eval_results.md"

if [ -z "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="~/.cache/huggingface/hub"
fi

eval_args="\
    --eval_name msmarco \
    --dataset_dir ./msmarco/data \
    --dataset_names $dataset_names \
    --splits dev dl19 dl20 \
    --corpus_embd_save_dir $corpus_embd_save_dir \
    --output_dir $output_dir \
    --search_top_k 1000 \
    --cache_path $HF_HUB_CACHE \
    --overwrite True \
    --k_values 10 100 1000 \
    --eval_output_method markdown \
    --eval_output_path $eval_output_path \
    --eval_metrics ndcg_at_10 recall_at_1000 mrr_at_10 \
"

model_args="\
    --embedder_name_or_path $model_name_or_path \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
    --cache_dir $HF_HUB_CACHE \
    --embedder_query_max_length 512 \
    --embedder_batch_size 256 \
"

cmd="python -m FlagEmbedding.evaluation.msmarco \
    $eval_args \
    $model_args \
"

# echo $cmd
# eval $cmd