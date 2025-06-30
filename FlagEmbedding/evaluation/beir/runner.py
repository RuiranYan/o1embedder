import logging
from FlagEmbedding.abc.evaluation import AbsEvalRunner

from .data_loader import BEIREvalDataLoader
from .prompts import BEIRInstructions
from .evaluator import BEIREvaluator
from .arguments import BEIREvalArgs, BEIREvalModelArgs

from FlagEmbedding.inference.embedder.decoder_only.base import BaseLLMEmbedder
from FlagEmbedding.inference.embedder.decoder_only.modeling_o1embedder import O1LLMEmbedder

from FlagEmbedding.inference.embedder.model_mapping import (
    EmbedderModelClass,
    AUTO_EMBEDDER_MAPPING, EMBEDDER_CLASS_MAPPING
)
from FlagEmbedding import FlagAutoModel, FlagAutoReranker

import os
import json

logger = logging.getLogger(__name__)


class BEIREvalRunner(AbsEvalRunner):
    def run(self):
        if self.eval_args.dataset_names is None:
            dataset_names = self.data_loader.available_dataset_names()
        else:
            dataset_names = self.data_loader.check_dataset_names(self.eval_args.dataset_names)

        if len(dataset_names) == 0:
            logger.info(f"Running {self.eval_args.eval_name} evaluation on the default dataset.")
            self.evaluator(
                splits=self.eval_args.splits,
                search_results_save_dir=self.eval_args.output_dir,
                retriever=self.retriever,
                reranker=self.reranker,
                corpus_embd_save_dir=self.eval_args.corpus_embd_save_dir,
                ignore_identical_ids=self.eval_args.ignore_identical_ids,
                k_values=self.eval_args.k_values
            )
            logger.info(f"{self.eval_args.eval_name} evaluation completed.")
        else:
            logger.info(f"Running {self.eval_args.eval_name} evaluation on the following dataset names: {dataset_names}")
            for dataset_name in dataset_names:
                logger.info(f"Running {self.eval_args.eval_name} evaluation on: {dataset_name}")
                self.evaluator(
                    splits=self.eval_args.splits,
                    search_results_save_dir=self.eval_args.output_dir,
                    retriever=self.retriever,
                    reranker=self.reranker,
                    corpus_embd_save_dir=self.eval_args.corpus_embd_save_dir,
                    ignore_identical_ids=self.eval_args.ignore_identical_ids,
                    k_values=self.eval_args.k_values,
                    dataset_name=dataset_name,
                )
            logger.info(f"{self.eval_args.eval_name} evaluation on {dataset_names} completed.")

        logger.info("Start computing metrics.")
        self.evaluate_metrics(
            search_results_save_dir=self.eval_args.output_dir,
            output_method=self.eval_args.eval_output_method,
            output_path=self.eval_args.eval_output_path,
            metrics=self.eval_args.eval_metrics
        )
        
    def load_data_loader(self) -> BEIREvalDataLoader:
        data_loader = BEIREvalDataLoader(
            eval_name=self.eval_args.eval_name,
            dataset_dir=self.eval_args.dataset_dir,
            cache_dir=self.eval_args.cache_path,
            token=self.eval_args.token,
            force_redownload=self.eval_args.force_redownload,
        )
        return data_loader

    def load_evaluator(self) -> BEIREvaluator:
        evaluator = BEIREvaluator(
            eval_name=self.eval_args.eval_name,
            data_loader=self.data_loader,
            overwrite=self.eval_args.overwrite,
        )
        return evaluator
    
    @staticmethod
    def get_models(model_args: BEIREvalModelArgs):
        # todo:
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
