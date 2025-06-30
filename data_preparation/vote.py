import json
import os
import sys
import time
from tqdm import tqdm
from pathlib import Path
import argparse

from transformers import AutoTokenizer, AutoModel
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
import math
from rank_bm25 import BM25Okapi
import numpy as np
import ast

# from .utils import last_token_pool, cal_score_dense, cal_score_bm25_by_rank_bm25
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, default=None)
parser.add_argument("--output_file", type=str, default=None)
parser.add_argument("--model_zoo", type=str, default=None, help='List of model names')
args = parser.parse_args()

raw_data_path = args.input_file
output_path = args.output_file
model_zoo = model_list = ast.literal_eval(args.model_zoo)

def cal_score_dense(syn_list: List[str], gth: List[str], model: SentenceTransformer, prompt_name: str =None) -> List[float]:
    model.max_seq_length = 512
    gth = gth[:1]
    thought_embeddings = model.encode(syn_list, normalize_embeddings=True)
    if prompt_name:
        doc_embeddings = model.encode(gth, prompt_name=prompt_name, normalize_embeddings=True)
    else:
        doc_embeddings = model.encode(gth, normalize_embeddings=True)
    scores = (doc_embeddings @ thought_embeddings.T).squeeze() * 100
    return scores.tolist()

def cal_score_bm25_by_rank_bm25(syn_list: List[str], gth: List[str]) -> List[float]:
    """计算 BM25 得分，使用 rank_bm25 库"""
    # 将 ground truth 列表转换为词列表
    query = gth[0].strip().lower().split(" ")
    doc = [syn.strip().lower().split(" ") for syn in syn_list]

    # 初始化 BM25 模型
    bm25 = BM25Okapi(doc)

    scores = bm25.get_scores(query)

    return scores.tolist()


# def split_response(text: str) -> List[str]:
#     # 切分文本
#     segments = [segment.strip() for segment in text.split('[end]') if segment.strip()]
#     # 去掉开头的 [begin] 标签
#     segments = [segment.replace('[begin]', '').strip() for segment in segments]
#     return segments

def vote(scores: List[List[float]], weight: List[float], voting="soft") -> int:
    # scores: [[model1_score1, model1_score2, ...], [model2_score1, model2_score2, ...], ...]
    assert voting in ["soft", "hard", "middle"]
    if voting == "soft":
        # 投票，分越高越好
        try:
            scores = torch.tensor(scores)
        except:
            print(scores)
        # normalize
        # print(scores.shape)
        assert scores.shape[1] == 3, "scores 的维度不正确"+str(scores.shape)
        scores = F.softmax(scores, dim=1)
        # weighted sum
        vote_score = torch.sum(scores * torch.tensor(weight).unsqueeze(1), dim=0)
        return torch.argmax(vote_score).item()
    elif voting == "hard":
        # 多数表决，聚合所有模型分最高的类别，如果有多个，取第一个，即不考虑weight
        # 将每个模型的分数转换为排名（选择分数最高的类别）
        try:
            scores = torch.tensor(scores)
        except:
            print(scores)
            
        # 获取每个模型预测的最高分类别
        max_indices = torch.argmax(scores, dim=1)
        
        # 统计每个类别的票数
        vote_counts = torch.bincount(max_indices)
        # print(vote_counts) [0,2,2,0]
        # 找出得票最多的类别索引
        max_votes = torch.max(vote_counts)
        # print(max_votes) 2
        # print(torch.nonzero(vote_counts == max_votes)) tensor([[1], [2]])
        winners = torch.nonzero(vote_counts == max_votes)
        # print(winners) tensor([[1], [2]])
        
        # 如果有多个类别得票相同,返回第一个
        return winners[0][0].item()
    elif voting == "middle":
        try:
            scores = torch.tensor(scores)
        except:
            print(scores)
        # normalize
        # print(scores.shape)
        assert scores.shape[1] == 7, "scores 的维度不正确"+str(scores.shape)
        scores = F.softmax(scores, dim=1)
        # weighted sum
        vote_score = torch.sum(scores * torch.tensor(weight).unsqueeze(1), dim=0)
        # 取中间值所在的idx
        middle_idx = len(vote_score) // 2
        return torch.argsort(vote_score)[middle_idx].item()

def main(raw_data_path: str, output_path: str, model_zoo: List[str], prompt_name_list: List[str], weight: List[float]) -> None:
    # 读取原始数据
    with open(raw_data_path, "r") as f:
        data = [json.loads(line) for line in f]

    # 初始化模型和分词器
    # model_list = [AutoModel.from_pretrained(model_name) for model_name in model_zoo]
    # tokenizer_list = [AutoTokenizer.from_pretrained(model_name) for model_name in model_zoo]
    # for i in range(len(model_list)):
    #     model_list[i].to('cuda')
    #     model_list[i].eval()
    print("loading retrieval models...")
    model_list = [SentenceTransformer(model_name,trust_remote_code=True) for model_name in model_zoo]

    # print('data:', data)

    # 逐条处理数据
    print("voting...")
    with open(output_path, "w") as f:
        for item in tqdm(data):
            model2score = {}
            query = item["query"]
            pos = item["pos"]
            response = item["all_thought"]
            # 判断 response 是否为Error:开头
            # if response.startswith("Error:") or response.startswith("错误"):
            #     print("query:\n", query)
            #     print(response)
            #     continue
            try:
            # 分割 response
                syn_list = response
                # syn_list = split_response(response)
                # 计算分数
                all_scores = []
                for i in range(len(model_list)):
                    scores_dense = cal_score_dense(syn_list, pos, model_list[i])
                    all_scores.append(scores_dense)
                    model2score[model_zoo[i]] = scores_dense
                scores_bm25 = cal_score_bm25_by_rank_bm25(syn_list, pos)
                all_scores.append(scores_bm25) # 保存了所有模型的分数
                model2score["bm25"] = scores_bm25
                vote_idx = vote(all_scores, weight=weight, voting="soft")
                vote_result = syn_list[vote_idx]
                # 保存结果
                item["thought"] = vote_result
                item["model2scores"] = model2score
                f.write(json.dumps(item) + "\n")
            except Exception as e:
                print(e)
                print("Error: ")
                print("syn_list:", syn_list)
                print("pos:", pos)
                continue

prompt_name_list = [ "None", "s2s_query", "None"]
weight = [1.0] * (len(model_zoo) + 1)
main(raw_data_path, output_path, model_zoo, prompt_name_list, weight=weight)
