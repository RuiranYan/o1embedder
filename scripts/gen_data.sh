python data_preparation/gen_gpt.py \
--log_path dataset/gen_llama-3-1-70b-2024.log \
--input_file dataset/toy.jsonl \
--output_file dataset/toy_thought.jsonl \
--checkpoint_file dataset/checkpoint_llama-3-1-70b.json \
--api_key EMPTY \
--base_url http://localhost:8000/v1/ \
--model_name Llama-3-1-70B-Instruct \
--task_description "Think about a plausible response to address the query." 