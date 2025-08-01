import logging
import os
import mteb
import json
import pandas as pd
from typing import Tuple, Union

from FlagEmbedding.abc.evaluation import AbsEvalRunner, AbsEvalModelArgs

from .arguments import MTEBEvalArgs, MTEBEvalModelArgs
from .searcher import MTEBEvalDenseRetriever, MTEBEvalReranker
from .prompts import get_task_def_by_task_name_and_type
from  .examples import examples_dict

from FlagEmbedding.inference.embedder.decoder_only.base import BaseLLMEmbedder
from FlagEmbedding.inference.embedder.decoder_only.modeling_o1embedder import O1LLMEmbedder

logger = logging.getLogger(__name__)


class MTEBEvalRunner(AbsEvalRunner):
    def __init__(
        self,
        eval_args: MTEBEvalArgs,
        model_args: MTEBEvalModelArgs,
    ):
        self.eval_args = eval_args
        self.model_args = model_args

        self.retriever, self.reranker = self.load_retriever_and_reranker()

    def load_retriever_and_reranker(self) -> Tuple[MTEBEvalDenseRetriever, Union[MTEBEvalReranker, None]]:
        embedder, reranker = self.get_models(self.model_args)
        retriever = MTEBEvalDenseRetriever(
            embedder,
            search_top_k=self.eval_args.search_top_k,
            overwrite=self.eval_args.overwrite
        )
        if reranker is not None:
            reranker = MTEBEvalReranker(reranker, rerank_top_k=self.eval_args.rerank_top_k)
        return retriever, reranker

    def read_results(self, output_folder, tasks):
        tasks_results = {}
        task_types = list(set([t.metadata.type for t in tasks]))
        for t_type in task_types:
            tasks_results[t_type] = {}
            for t in tasks:
                if t.metadata.type != t_type: continue
                task_name = t.metadata.name

                metric = t.metadata.main_score
                split = t.metadata.eval_splits[0]

                if os.path.exists(os.path.join(output_folder, task_name + '.json')):
                    data = json.load(open(os.path.join(output_folder, task_name + '.json')))
                    tasks_results[t_type][task_name] = {}
                    for s in ['test', 'dev', 'validation', 'standard']:
                        if s in data['scores']:
                            split = s
                            break
                        split = None
                    if split is None:
                        print('ERROR')
                        break

                    temp_data = data['scores'][split][0]

                    if metric == 'ap':
                        tasks_results[t_type][task_name] = round(temp_data['cos_sim']['ap'] * 100, 2)
                    elif metric == 'cosine_spearman':
                        tasks_results[t_type][task_name] = round(temp_data['cos_sim']['spearman'] * 100, 2)
                    else:
                        tasks_results[t_type][task_name] = round(temp_data[metric] * 100, 2)
        print(f"tasks_results: {tasks_results}")
        return tasks_results

    def output_json(self, tasks_results, save_file):
        all_results = 0
        all_results_num = 0
        cqa_results = 0
        cqa_results_num = 0

        new_results = {}
        for task_type in tasks_results.keys():
            new_results[task_type] = {}
            tmp_results = 0
            tmp_results_num = 0
            for task_name in tasks_results[task_type].keys():
                if "CQADupstack" in task_name:
                    cqa_results += tasks_results[task_type][task_name]
                    cqa_results_num += 1
                else:
                    new_results[task_type][task_name] = float(round(tasks_results[task_type][task_name], 2))
                    all_results_num += 1
                    all_results += tasks_results[task_type][task_name]
                    tmp_results_num += 1
                    tmp_results += tasks_results[task_type][task_name]
            if cqa_results_num > 0:
                cqa_results = cqa_results / cqa_results_num
                new_results[task_type]["CQADupstack"] = float(round(cqa_results, 2))
                all_results += cqa_results
                all_results_num += 1
                tmp_results += cqa_results
                tmp_results_num += 1
            new_results[task_type]['Avg'] = float(round(tmp_results / tmp_results_num, 2))
        new_results['Avg'] = float(round(all_results / all_results_num, 2))
        with open(save_file, 'w') as f:
            json.dump(new_results, f)

    def run(self):
        task_types = self.eval_args.task_types
        tasks = self.eval_args.tasks
        languages = self.eval_args.languages
        tasks = mteb.get_tasks(
            languages=languages,
            tasks=tasks,
            task_types=task_types
        )
        output_folder = self.eval_args.output_dir
        new_tasks = []
        for task in tasks:
            if task.languages is not None:
                if len(task.languages) == len([e for e in languages if e in task.languages]):
                    new_tasks.append(task)

        for task in new_tasks:
            task_name = task.metadata.name
            task_type = task.metadata.type

            if self.eval_args.use_special_instructions:
                try:
                    instruction = get_task_def_by_task_name_and_type(task_name, task_type)
                    self.retriever.set_instruction(instruction)
                except:
                    logger.logger.info(f"No instruction found for {task_name}")

            if self.eval_args.use_special_examples:
                try:
                    eg_pairs = examples_dict[task_name]
                    self.retriever.set_examples(eg_pairs)
                except:
                    logger.logger.info(f"No examples found for {task_name}")

            if task_type == 'Classification':
                self.retriever.set_normalize_embeddings(False)
            else:
                self.retriever.set_normalize_embeddings(True)

            evaluation = mteb.MTEB(tasks=[task])
            results = evaluation.run(self.retriever, output_folder=f"{output_folder}/{str(self.retriever)}")

        logger.info("Start computing metrics. Only save results as json.")
        tasks_results = self.read_results(f"{output_folder}/{str(self.retriever)}/no_model_name_available/no_revision_available", new_tasks)
        self.output_json(tasks_results, self.eval_args.eval_output_path)

    # hack 
    @staticmethod
    def get_models(model_args: MTEBEvalModelArgs):
        retriever = O1LLMEmbedder(
            model_name_or_path=model_args.embedder_name_or_path,
            normalize_embeddings=model_args.normalize_embeddings,
            use_fp16=model_args.use_fp16,
            # query_instruction_for_retrieval=model_args.query_instruction_for_retrieval,
            # query_instruction_format=model_args.query_instruction_format_for_retrieval,
            query_instruction_for_retrieval=None, # for now, only support this format
            query_instruction_format="{}{}", # for now, only support this format
            devices=model_args.devices,
            trust_remote_code=model_args.trust_remote_code,
            cache_dir=model_args.cache_dir,
            batch_size=model_args.embedder_batch_size,
            query_max_length=model_args.embedder_query_max_length,
            passage_max_length=model_args.embedder_passage_max_length,
            last_with_query=model_args.last_with_query,
            n_ans=model_args.n_ans,
        )
        return retriever, None