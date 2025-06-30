vllm serve /home/yanruiran/.cache/modelscope/hub/LLM-Research/Meta-Llama-3___1-70B-Instruct/ \
  --served-model-name Llama-3-1-70B-Instruct \
  --max-model-len 8192 \
  --tensor-parallel-size 8 \
  --enforce-eager \
  --gpu-memory-utilization 0.95
