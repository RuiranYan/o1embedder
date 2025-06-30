import json
from tqdm import tqdm

from typing import List, Dict, Mapping
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

import time
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
import random


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=192)
parser.add_argument("--max_workers", type=int, default=192)
parser.add_argument("--log_path", type=str, default="generation.log")
parser.add_argument("--input_file", type=str, default=None)
parser.add_argument("--output_file", type=str, default=None)
parser.add_argument("--checkpoint_file", type=str, default="checkpoint_generation.json")
parser.add_argument("--task_description", type=str, default="Write a passage that answers the given query.")
parser.add_argument("--api_key", type=str, default=None)
parser.add_argument("--base_url", type=str, default=None)
parser.add_argument("--model_name", type=str, default="gpt-4o-mini")
parser.add_argument("--thought_num", type=int, default=3)
args = parser.parse_args()


raw_data_path = args.input_file
output_response_path = args.output_file
checkpoint_path = args.checkpoint_file
task_description = args.task_description
api_key = args.api_key
base_url = args.base_url
model_name = args.model_name
thought_num = args.thought_num

PROMPT_TEMPLATE ="""\
Task: 
{task}

Examples: 
{example}

Query: 
{query}
Response: \
"""


# model_name = "Llama-3-1-70B-Instruct"
# api_key = "empty"
# base_url = "http://localhost:8000/v1/"

def build_example(data: List, num: int) -> str:
    # random choice num data, build example like:
    # Query: data['query']
    # Response: data['pos'][0]
    examples = random.choices(data, k=num)
    examples = "\n".join([f"Query: {d['query']}\nResponse: {d['pos'][0]}" for d in examples])
    return examples


client = OpenAI(api_key=api_key, base_url=base_url)

# 添加日志配置
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(args.log_path),
        logging.StreamHandler()
    ]
)

def generate_one(client: OpenAI, prompt, model="gpt-4o-mini", max_retries=3, retry_delay=1):
    for attempt in range(max_retries):
        # try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                # {"role": "system", "content": "You are a helpful assistant. Your anwer should be concise and follow the task description."},
                # {"role": "system", "content": "You are asked to write a passage that answers the given query. Do not ask the user for further clarification. Your answer should be concise and informative. Don't repeat the query, just give me the response."},  
                {"role": "system", "content": "You are a helpful assistant. Your anwer should be follow the task description. Do not ask the user for further clarification. Don't repeat the query, just give the response."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
            timeout=30,  # 添加超时设置
        )
        content = response.choices[0].message.content
                
        return content.strip()
            
        # except Exception as e:
        #     logging.warning(f"尝试 {attempt + 1}: {str(e)}")
        #     if attempt == max_retries - 1:
        #         # return f"错误: {str(e)}"
        #         logging.warning(f"错误: {str(e)}")
        #         return ""
        #     time.sleep(retry_delay * (2 ** attempt))

def process_batch(batch_prompts, batch_data, client, model):
    batch_results = []
    futures_to_data = {}
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for idx, prompt in enumerate(batch_prompts):
            future = executor.submit(generate_one, client, prompt, model)
            futures_to_data[future] = (idx, batch_data[idx])
        
        for future in as_completed(futures_to_data):
            idx, current_data = futures_to_data[future]
            try:
                result = future.result()
                # print("\n\n result:", result)
                current_data = current_data.copy()
                current_data["all_thought"] = result
                # current_data["query_label"] = result
                # current_data["timestamp"] = datetime.now().isoformat()
                batch_results.append((idx, current_data))
            except Exception as e:
                logging.error(f"批次处理错误 (idx={idx}): {str(e)}")
                current_data = current_data.copy()
                # current_data["response"] = f"错误: {str(e)}"
                logging.warning(f"错误: {str(e)}")
                current_data["all_thought"] = ""
                # current_data["timestamp"] = datetime.now().isoformat()
                batch_results.append((idx, current_data))
    
    # 按原始顺序排序结果
    return [item[1] for item in sorted(batch_results, key=lambda x: x[0])]

def generate_parallel(client, prompts, data, model="gpt-4o-mini", num_rounds=3):
    start_idx = 0

    Path(output_response_path).parent.mkdir(parents=True, exist_ok=True)

    # if os.path.exists(checkpoint_path):
    #     with open(checkpoint_path, 'r') as f:
    #         checkpoint_data = json.load(f)
    #         start_idx = checkpoint_data.get('last_processed_idx', 0)
    #         logging.info(f"从检查点恢复: 索引 {start_idx}")

    remaining_prompts = prompts[start_idx:]
    remaining_data = data[start_idx:]

    try:
        for batch_start in tqdm(range(0, len(remaining_data), args.batch_size)):
            # batch_end = min(batch_start + args.batch_size, len(remaining_data))
            last_batch_size = -1
            if batch_start + args.batch_size > len(remaining_data):
                last_batch_size = len(remaining_data) - batch_start
                batch_end = len(remaining_data)
            else:
                batch_end = batch_start + args.batch_size
            batch_start2 = int(num_rounds * batch_start)
            batch_end2 = min(batch_start2 + args.batch_size * num_rounds, len(remaining_prompts))
            batch_prompts = remaining_prompts[batch_start2:batch_end2]
            batch_data = remaining_data[batch_start:batch_end]

            all_responses = [[] for _ in range(len(batch_data))]
            # all_labels = [[] for _ in range(len(batch_data))]

            for round_idx in range(num_rounds):
                if last_batch_size > 0:
                    round_batch_prompts = batch_prompts[round_idx * last_batch_size : (round_idx + 1) * last_batch_size]
                else:
                    round_batch_prompts = batch_prompts[round_idx * args.batch_size : (round_idx + 1) * args.batch_size]
                batch_results = process_batch(round_batch_prompts, batch_data, client, model)
                for i, res in enumerate(batch_results):
                    all_responses[i].append(res["all_thought"])
                    # all_labels[i].append(res.get("query_label", f"label_{round_idx}"))  # 或根据需要生成 label

            merged_results = []
            for i, d in enumerate(batch_data):
                d = d.copy()
                d["all_thought"] = all_responses[i]
                # d["query_label"] = all_labels[i]
                # d["timestamp"] = datetime.now().isoformat()
                merged_results.append(d)

            with open(output_response_path, 'a') as f:
                for item in merged_results:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            current_idx = start_idx + batch_end
            with open(checkpoint_path, 'w') as f:
                checkpoint_data = {
                    'last_processed_idx': current_idx,
                    # 'timestamp': datetime.now().isoformat(),
                    'total_samples': len(data)
                }
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)

            logging.info(f"批次完成: {current_idx}/{len(data)} ({(current_idx/len(data)*100):.2f}%)")

    except KeyboardInterrupt:
        logging.info("检测到用户中断，正在保存进度...")
        with open(checkpoint_path, 'w') as f:
            json.dump({'last_processed_idx': current_idx}, f)
        sys.exit(1)


# 生成数据
if os.path.exists(output_response_path):
    print(f"文件 {output_response_path} 已存在,将从上次位置继续生成...")

# 读取数据
with open(raw_data_path, "r") as f:
    data = [json.loads(line) for line in tqdm(f)]

# 生成提示
# prompts = [PROMPT_TEMPLATE.format(task=d["prompt"], example=build_example(data, 3), query=d["query"]) for d in data]
# prompts = [PROMPT_TEMPLATE.format(task=task_description, example=build_example(data, 3), query=d["query"]) for d in data]

print("build prompts...")
prompts = [] 
for t in range(thought_num):
    for d in tqdm(data):
        example=build_example(data, 3)
        # filter out too long example
        if len(example) > 4096 * 5:
            example = example[:4096 * 5]
        p = PROMPT_TEMPLATE.format(task=task_description, example=example, query=d["query"])
        prompts.append(p)
print("提示数量:", len(prompts))
print("示例提示:")
print(prompts[0])

# 生成响应
print("开始生成响应...")
generate_parallel(client, prompts, data, model=model_name, num_rounds=thought_num)

print("生成完成")

# 删除checkpoint
if os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)
