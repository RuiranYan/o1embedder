from FlagEmbedding.abc.evaluation import AbsEvalRunner

from .data_loader import MSMARCOEvalDataLoader
from .arguments import MSMARCOEvalArgs, MSMARCOEvalModelArgs
# from .modeling_o1embedder import O1LLMEmbedder
from .repllama import RepllamaLLMEmbedder
from FlagEmbedding.inference.embedder.decoder_only.base import BaseLLMEmbedder
from FlagEmbedding.inference.embedder.decoder_only.modeling_o1embedder import O1LLMEmbedder

from FlagEmbedding.inference.embedder.model_mapping import (
    EmbedderModelClass,
    AUTO_EMBEDDER_MAPPING, EMBEDDER_CLASS_MAPPING
)
from FlagEmbedding import FlagAutoModel, FlagAutoReranker


import datasets
import os 
import json

class MSMARCOEvalRunner(AbsEvalRunner):
    def load_data_loader(self) -> MSMARCOEvalDataLoader:
        data_loader = MSMARCOEvalDataLoader(
            eval_name=self.eval_args.eval_name,
            dataset_dir=self.eval_args.dataset_dir,
            cache_dir=self.eval_args.cache_path,
            token=self.eval_args.token,
            force_redownload=self.eval_args.force_redownload,
        )
        return data_loader
    
    
    @staticmethod
    def get_models(model_args: MSMARCOEvalModelArgs):
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
            q2r_dic_path=model_args.q2r_dic_path,
            last_with_query=model_args.last_with_query,
            n_ans=model_args.n_ans,
        )
        return retriever, None
