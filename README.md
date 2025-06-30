# 🧠 O1 Embedder: Enhanced Retrieval Model with Thinking Capabilities

## 🔍 Overview

O1 Embedder is a **reasoning-enhanced dense retriever** that mimics the step-by-step thinking behavior of Large Language Models (LLMs) to solve complex and zero-shot retrieval tasks.

It is the **first retrieval model that integrates long-form thought generation and discriminative embedding** in a unified framework — enabling high performance on both in-domain and out-of-distribution (OOD) information retrieval benchmarks.


## ✨ Key Features

- **🧠 Thought-Augmented Retrieval**: Generates LLM-style "thoughts" before embedding the query to uncover hidden semantic intents.
- **🔁 Joint Multi-task Training**: Simultaneous optimization for generation and retrieval via behavior cloning & contrastive learning.
- **📊 Strong Generalization**: Achieves SoTA or near-SoTA results on 12 retrieval benchmarks including MS MARCO, HotPotQA, SciFact, CosQA.
- **🧪 Backbone-Agnostic**: Compatible with LLaMA, Mistral, Qwen, and other major open-source LLMs.

## 🏁 Quick Start

### 1. Clone this repo

```bash
git clone https://github.com/your-org/O1-Embedder.git
cd O1-Embedder
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install https://github.com/kyamagu/faiss-wheels/releases/download/v1.7.3/faiss_gpu-1.7.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

### 3. Download Datasets
```bash
huggingface-cli download --repo-type dataset --resume-download Ruiran/msmarco_thought final.jsonl --local-dir dataset --local-dir-use-symlinks False
```

### (Optional) Prepare Your Own Training Data 

You can build your own thought-augmented dataset via:

1. Start the vLLM server:
```bash
bash ./scripts/vllm.sh
```
2. Open another terminal and access the vLLM server:
```bash
bash ./scripts/gen_data.sh
```
3. Vote and get the best thought:
```bash
python ./data_preparation/vote.py --input_file "./dataset/toy_thought.jsonl" --output_file "./dataset/toy_vote_res.jsonl" --model_zoo '["BAAI/bge-large-en-v1.5", "dunzhang/stella_en_1.5B_v5", "Alibaba-NLP/gte-large-en-v1.5"]'
```


### 4. Train O1 Embedder

```bash
bash scripts/train.sh
```

### 5. Evaluation

```bash
bash scripts/eval.sh
```

## 🧠 Core Ideas

### 🧪 1. Thought Generation via LLM
Use a strong LLM (e.g., LLaMA-3) to generate long-form "thoughts" before retrieval.

### 🧪 2. Retrieval Committee
Evaluate thought quality via a diverse set of retrievers and select via majority voting.

### 🧪 3. Joint Learning
- **Behavior Cloning**: teaches the model to generate thoughts.
- **Contrastive Learning**: aligns query-thought pairs with relevant documents.


## 🤖 Supported Backbones

| Backbone     | Model Sizes | Supported |
|--------------|-------------|-----------|
| LLaMA        | 7B, 8B       | ✅        |
| Mistral      | 7B           | ✅        |
| Qwen2.5      | 0.5B–7B      | ✅        |

## 📝 Citation

If you find our work helpful, please cite our paper:

```
@misc{yan2025o1embedderletretrievers,
      title={O1 Embedder: Let Retrievers Think Before Action}, 
      author={Ruiran Yan and Zheng Liu and Defu Lian},
      year={2025},
      eprint={2502.07555},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.07555}, 
}
```
