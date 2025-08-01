from tqdm import tqdm
from typing import cast, Any, List, Union, Optional, Dict

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

from FlagEmbedding.abc.inference import AbsEmbedder
import os   
from peft import PeftModel,PeftConfig, PeftModelForCausalLM
import json


# Pooling function for LLM-based embedding models
def last_token_pool(last_hidden_states: torch.Tensor,
                    input_batches: Dict) -> torch.Tensor:
    attention_mask = input_batches['attention_mask']
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
def truncate_by_tokenizer(tokenizer, text, max_len):
    # Truncate the text if it is longer than max_len
    # The text is a string, return a string
    tokenized_text = tokenizer(text, truncation=True, max_length=max_len, return_tensors="pt")
    return tokenizer.decode(tokenized_text['input_ids'][0], skip_special_tokens=True)

class O1LLMEmbedder(AbsEmbedder):
    def __init__(
        self,
        model_name_or_path: str,
        normalize_embeddings: bool = True,
        use_fp16: bool = True,
        query_instruction_for_retrieval: str = None,
        query_instruction_format: str = "Instruct: {}\nQuery: {}", # specify the format of query_instruction_for_retrieval
        devices: Union[str, List[str]] = None, # specify devices, such as "cuda:0" or ["cuda:0", "cuda:1"]
        # Additional parameters for BaseLLMEmbedder
        trust_remote_code: bool = False,
        cache_dir: str = None,
        # inference
        batch_size: int = 256,
        query_max_length: int = 512,
        passage_max_length: int = 512,
        convert_to_numpy: bool = True,
        last_with_query: bool = False,
        n_ans: int = 3,
        **kwargs: Any,
    ):
        super().__init__(
            model_name_or_path,
            normalize_embeddings=normalize_embeddings,
            use_fp16=use_fp16,
            query_instruction_for_retrieval=query_instruction_for_retrieval,
            query_instruction_format=query_instruction_format,
            devices=devices,
            batch_size=batch_size,
            query_max_length=query_max_length,
            passage_max_length=passage_max_length,
            convert_to_numpy=convert_to_numpy,
            **kwargs
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir
        )
        self.tokenizer.padding_side = 'left'
        print(self.tokenizer.additional_special_tokens)
        if os.path.exists(os.path.join(model_name_or_path, "adapter_model.safetensors")):
            print("load lora")
            # load lora
            config = PeftConfig.from_pretrained(model_name_or_path)
            if config.task_type == "FEATURE_EXTRACTION":
                print("lora task type: feature extraction")
                self.model = AutoModel.from_pretrained(config.base_model_name_or_path, trust_remote_code=trust_remote_code, cache_dir=cache_dir)
                if 'embedding' in os.listdir(model_name_or_path):
                    self.model.set_input_embeddings(torch.load(os.path.join(model_name_or_path, 'embedding', 'emb.pth')))
                    print("reset embedding")
                self.model = PeftModel.from_pretrained(self.model, model_name_or_path, is_trainable=False)
                self.model = self.model.merge_and_unload() 
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                print(111)
            elif config.task_type == "CAUSAL_LM":
                print("lora task type: causal lm")
                self.model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, trust_remote_code=trust_remote_code, cache_dir=cache_dir)
                if 'embedding' in os.listdir(model_name_or_path):
                    self.model.set_input_embeddings(torch.load(os.path.join(model_name_or_path, 'embedding', 'emb.pth')))
                    self.model.set_output_embeddings(torch.load(os.path.join(model_name_or_path, 'embedding', 'lm_head.pth')))
                self.model = PeftModelForCausalLM.from_pretrained(self.model, model_name_or_path, is_trainable=False)
                self.model_lm = self.model.merge_and_unload() 
                self.model = self.model_lm.model # base model MistralModel
        else:
            self.model_lm = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=trust_remote_code,
                cache_dir=cache_dir
            )
            
            self.model = self.model_lm.model


        self.n_ans = n_ans

        if self.kwargs.get("pooling_method", "last_token") != "last_token":
            raise ValueError("Pooling method must be 'last_token' for LLM-based models.")


        self.last_with_query = last_with_query

    def last_token_pool(self, last_hidden_states: torch.Tensor,
                        input_batches: Dict) -> torch.Tensor:
        return last_token_pool(last_hidden_states, input_batches)
    
    def last_with_query_pool(self, last_hidden_states: torch.Tensor,
                            input_batches: Dict) -> torch.Tensor:
        batch_size = last_hidden_states.shape[0]
        query_end_token = self.tokenizer.convert_tokens_to_ids('</query>')
        if query_end_token in input_batches['input_ids']:
            query_end_positions = (input_batches['input_ids'] == query_end_token).nonzero()[:, 1]
            query_token_reps = last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), query_end_positions]
            return last_hidden_states[:, -1] + query_token_reps
            # return query_token_reps
            # return last_hidden_states[:, -1]
        else:
            return last_hidden_states[:, -1]
    
    def generate_thought(self, queries: List[str], n_ans: int = 3, device: str = None, batch_size: int = 16):
        """
        批量生成thought
        :param queries: 输入的query列表
        :param n_ans: 每个query生成的答案数量
        :param device: 推理设备
        :param batch_size: 批处理大小
        :return: List[List[str]]，每个query对应n_ans个thought
        """
        TEMPLATE_SPECIAL = "<query>{query}</query><thought>"
        EMB_TOKEN = "<emb>"
        results = []
        total = len(queries)
        for start_idx in tqdm(range(0, total, batch_size), desc="generate thought"):
            batch_queries = queries[start_idx:start_idx + batch_size]
            batch_prompts = [TEMPLATE_SPECIAL.format(query=q) for q in batch_queries]
            input_text = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
            outputs = self.model_lm.generate(
                **input_text,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                num_return_sequences=n_ans,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            # outputs shape: (batch_size * n_ans, seq_len)
            batch_decoded_outputs = []
            for i in range(len(batch_queries)):
                current_outputs = outputs[i * n_ans:(i + 1) * n_ans]
                current_decoded = []
                for output in current_outputs:
                    text = self.tokenizer.decode(output, skip_special_tokens=False)
                    # 去除pad_token
                    if self.tokenizer.pad_token is not None:
                        text = text.replace(self.tokenizer.pad_token, "")
                    # 保证结尾有eos_token
                    if self.tokenizer.eos_token is not None and not text.endswith(self.tokenizer.eos_token):
                        text = text + self.tokenizer.eos_token
                    current_decoded.append(text)
                batch_decoded_outputs.append(current_decoded)
            results.extend(batch_decoded_outputs)
        return results
    
    def encode_queries(
        self,
        queries: Union[List[str], str],
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        convert_to_numpy: Optional[bool] = None,
        **kwargs: Any
    ) -> Union[np.ndarray, torch.Tensor]:
        
        print(queries[0])
        kwargs['is_query'] = True
        return super().encode_queries(
            queries,
            batch_size=batch_size,
            max_length=max_length,
            convert_to_numpy=convert_to_numpy,
            **kwargs
        )

    def encode_corpus(
        self,
        corpus: Union[List[str], str],
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        convert_to_numpy: Optional[bool] = None,
        **kwargs: Any
    ) -> Union[np.ndarray, torch.Tensor]:

        print(corpus[0])
        return super().encode_corpus(
            corpus,
            batch_size=batch_size,
            max_length=max_length,
            convert_to_numpy=convert_to_numpy,
            **kwargs
        )

    def encode(
        self,
        sentences: Union[List[str], str],
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        convert_to_numpy: Optional[bool] = None,
        **kwargs: Any
    ) -> Union[np.ndarray, torch.Tensor]:
        return super().encode(
            sentences,
            batch_size=batch_size,
            max_length=max_length,
            convert_to_numpy=convert_to_numpy,
            **kwargs
        )

    @torch.no_grad()
    def encode_single_device(
        self,
        sentences: Union[List[str], str],
        batch_size: int = 256,
        max_length: int = 512,
        convert_to_numpy: bool = True,
        device: str = None,
        **kwargs: Any   # add `pad_to_multiple_of=8` for bge-multilingual-gemmma2
    ):
        # print("convert_to_numpy:", convert_to_numpy) # True
        is_query = kwargs.pop('is_query', False)
        if device is None:
            device = self.target_devices[0]

        if device == "cpu": self.use_fp16 = False
        if self.use_fp16: self.model.half()

        self.model.to(device)
        self.model.eval()

        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True

        if is_query:
            if self.use_fp16: self.model_lm.half()
            self.model_lm.to(device)
            self.model_lm.eval()
            print(f"generate thought for {len(sentences)} queries...")
            # print(sentences[:2])
            sentences = self.generate_thought(sentences, self.n_ans, device)
            print(f"generated {len(sentences)} thoughts finished")
            # print(sentences[:2])

            # flatten
            sentences = [item for sublist in sentences for item in sublist]
            # # for noT
            # TEMPLATE_SPECIAL = "<query>{query}</query><thought>"
            # sentences = [TEMPLATE_SPECIAL.format(query=q) + self.tokenizer.eos_token for q in sentences]
            # self.n_ans = 1
            # print(sentences[0])

        # hack
        if hasattr(self.tokenizer, 'add_eos_token'):
            self.tokenizer.add_eos_token = False
        # hack test
        assert self.tokenizer('a')['input_ids'][-1] != self.tokenizer.eos_token_id
        emb_token_id = self.tokenizer.convert_tokens_to_ids('<emb>')
        # emb_token_id = self.tokenizer.convert_tokens_to_ids('</s>')

        # tokenize without padding to get the correct length
        all_inputs = []
        for start_index in range(0, len(sentences), batch_size):
            sentences_batch = sentences[start_index:start_index + batch_size]
            inputs_batch = self.tokenizer(
                sentences_batch,
                truncation=True,
                max_length=max_length,
                return_tensors=None, # hack
                **kwargs
            )
            # hack: add eos_token
            inputs_batch['input_ids'] = [x + [emb_token_id] for x in inputs_batch['input_ids']]
            inputs_batch['attention_mask'] = [x + [1] for x in inputs_batch['attention_mask']]

            inputs_batch = [{
                k: inputs_batch[k][i] for k in inputs_batch.keys()
            } for i in range(len(sentences_batch))]
            all_inputs.extend(inputs_batch)

        # sort by length for less padding
        length_sorted_idx = np.argsort([-len(x['input_ids']) for x in all_inputs]) # 降序
        all_inputs_sorted = [all_inputs[i] for i in length_sorted_idx]

        # adjust batch size
        flag = False
        max_length_inputs = self.tokenizer.pad(
            all_inputs_sorted[:1],
            padding=True,
            return_tensors='pt',
            **kwargs
        ).to(device)
        while flag is False:
            try:
                test_inputs_batch = {}
                for k, v in max_length_inputs.items():
                    test_inputs_batch[k] = v.repeat(batch_size, 1)
                last_hidden_state = self.model(**test_inputs_batch, return_dict=True).last_hidden_state
                if self.last_with_query:
                    embeddings = self.last_with_query_pool(last_hidden_state, test_inputs_batch)
                else:
                    embeddings = last_token_pool(last_hidden_state, test_inputs_batch)
                flag = True
            except RuntimeError as e:
                batch_size = batch_size * 3 // 4
            except torch.OutofMemoryError as e:
                batch_size = batch_size * 3 // 4

        # encode
        all_embeddings = []
        for start_index in tqdm(range(0, len(sentences), batch_size), desc="Inference Embeddings",
                                disable=len(sentences) < 256):
            inputs_batch = all_inputs_sorted[start_index:start_index + batch_size]
            inputs_batch = self.tokenizer.pad(
                inputs_batch,
                padding=True,
                return_tensors='pt',
                **kwargs
            ).to(device)
            last_hidden_state = self.model(**inputs_batch, return_dict=True).last_hidden_state
            if self.last_with_query:
                embeddings = self.last_with_query_pool(last_hidden_state, inputs_batch)
            else:
                embeddings = self.last_token_pool(last_hidden_state, inputs_batch)
            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            embeddings = cast(torch.Tensor, embeddings)

            if convert_to_numpy:
                embeddings = embeddings.cpu().numpy()
            all_embeddings.append(embeddings)

        if convert_to_numpy:
            all_embeddings = np.concatenate(all_embeddings, axis=0)
        else:
            all_embeddings = torch.cat(all_embeddings, dim=0)

        # adjust the order of embeddings
        all_embeddings = all_embeddings[np.argsort(length_sorted_idx)]

        # aggregate
        if is_query:
            if convert_to_numpy:
                all_embeddings = all_embeddings.reshape(-1, self.n_ans, all_embeddings.shape[-1])
                all_embeddings = np.mean(all_embeddings, axis=1)
            else:
                all_embeddings = all_embeddings.reshape(-1, self.n_ans, all_embeddings.shape[-1])
                all_embeddings = torch.mean(all_embeddings, dim=1)

        # return the embeddings
        if input_was_string:
            return all_embeddings[0]
        return all_embeddings


if __name__ == "__main__":
    model = O1LLMEmbedder(
        model_name_or_path="/share_2/ruiran/projects/O1Embedder/checkpoints/qwen2.5_7b_all/merged_model",
        devices=["cuda:0", "cuda:1"],
        n_ans=3
    )
    queries = ["What is the capital of France?"]*100
    q_emb = model.encode_queries(queries, batch_size=100, max_length=1024, convert_to_numpy=True)
    print(q_emb.shape)
    del model
