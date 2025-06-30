import math
import random
import logging
import torch
from dataclasses import dataclass
from transformers import (
    PreTrainedTokenizer, 
    DataCollatorWithPadding,
)

from FlagEmbedding.abc.finetune.embedder import AbsEmbedderSameDatasetTrainDataset

from .arguments import DecoderOnlyEmbedderO1DataArguments

logger = logging.getLogger(__name__)


IGNORE_INDEX=-100
EMB_TOKEN = "<emb>"
RESPONSE_TOKEN = "<thought>"
def truncate_by_tokenizer(text, tokenizer, max_len):
    # Truncate the text if it is longer than max_len
    # The text is a string, return a string
    tokenized_text = tokenizer(text, truncation=True, max_length=max_len, return_tensors="pt")
    return tokenizer.decode(tokenized_text['input_ids'][0], skip_special_tokens=True)
    

class DecoderOnlyEmbedderO1SameDatasetTrainDataset(AbsEmbedderSameDatasetTrainDataset):
    def __init__(
        self,
        args: DecoderOnlyEmbedderO1DataArguments,
        default_batch_size: int,
        seed: int,
        tokenizer: PreTrainedTokenizer,
        process_index: int=0,
        num_processes: int=1
    ):
        super().__init__(
            args=args,
            default_batch_size=default_batch_size,
            seed=seed,
            tokenizer=tokenizer,
            process_index=process_index,
            num_processes=num_processes
        )
        self.args: DecoderOnlyEmbedderO1DataArguments

    def __getitem__(self, _):
        batch_indices, no_in_batch_neg_flag = self.batch_datas[self.step]    # extend here
        cur_batch_size = int(len(batch_indices) / self.num_processes)
        batch_indices = batch_indices[self.process_index * cur_batch_size: (self.process_index + 1) * cur_batch_size]
        batch_data = self.dataset[batch_indices]
        self.step += 1
        queries, passages, thought, teacher_scores = self._create_batch_data(batch_raw_data=batch_data)
        # print([queries, passages, thought, teacher_scores, no_in_batch_neg_flag])
        return queries, passages, thought, teacher_scores, no_in_batch_neg_flag

    def _create_batch_data(self, batch_raw_data):
        queries, passages, teacher_scores = [], [], []

        train_group_size, data_type = self._get_train_group_size(batch_raw_data)
        thought = []

        for i in range(len(batch_raw_data['query'])):
            if data_type is not None:
                assert batch_raw_data['type'][i] == data_type, f"Data type is not consistent in the same batch"

            # truncate query
            batch_raw_data['query'][i] = truncate_by_tokenizer(text=batch_raw_data['query'][i], tokenizer=self.tokenizer, max_len=self.args.query_max_len) # need improvement

            
            # queries.append(
            #     self.args.query_instruction_format.format(
            #         batch_raw_data['prompt'][i] if 'prompt' in batch_raw_data else self.args.query_instruction_for_retrieval,
            #         batch_raw_data['query'][i]
            #     )
            # ) # --query_instruction_format '<instruct>{}\n<query>{}</query>\n<thought>'
            # queries.append(self.args.query_instruction_format.format("", batch_raw_data['query'][i])) # --query_instruction_format '<instruct>{}<query>{}</query><thought>' # no instruct
            queries.append(self.args.query_instruction_format.format(batch_raw_data['query'][i])) # --query_instruction_format '<query>{}</query>' # no instruct
            
            tmp_passages = []
            pos_idx = random.choice(list(range(len(batch_raw_data['pos'][i]))))
            pos = batch_raw_data['pos'][i][pos_idx]
            # pos = self._shuffle_text(batch_raw_data['pos'][i][pos_idx])
            tmp_passages.append(pos)

            neg_all_idx = list(range(len(batch_raw_data['neg'][i])))
            if len(batch_raw_data['neg'][i]) < train_group_size - 1:
                num = math.ceil((train_group_size - 1) / len(batch_raw_data['neg'][i]))
                neg_idxs = random.sample(neg_all_idx * num, train_group_size - 1)
            else:
                neg_idxs = random.sample(neg_all_idx, train_group_size - 1)
            for neg_idx in neg_idxs:
                tmp_passages.append(batch_raw_data['neg'][i][neg_idx])

            if self.args.knowledge_distillation:
                if 'pos_scores' in batch_raw_data and batch_raw_data['pos_scores'][i] is not None:
                    teacher_scores.append(batch_raw_data['pos_scores'][i][pos_idx])
                for neg_idx in neg_idxs:
                    if 'neg_scores' in batch_raw_data and batch_raw_data['neg_scores'][i] is not None:
                        teacher_scores.append(batch_raw_data['neg_scores'][i][neg_idx])
            else:
                teacher_scores = None

            if data_type is not None and data_type in ['symmetric_sts', 'symmetric_clustering']:
                tmp_passages = [
                    self.args.query_instruction_format.format(
                        batch_raw_data['prompt'][i] if 'prompt' in batch_raw_data else self.args.query_instruction_for_retrieval,
                        p
                    ) for p in tmp_passages
                ]
            else:
                if self.args.passage_instruction_for_retrieval is not None:
                    tmp_passages = [
                        self.args.passage_instruction_format.format(
                            self.args.passage_instruction_for_retrieval, p
                        ) for p in tmp_passages
                    ]


                if "thought" in batch_raw_data and batch_raw_data['thought'][i] != "":
                    thought.append(batch_raw_data['thought'][i] + self.tokenizer.eos_token)
                else:
                    thought.append("")

            passages.extend(tmp_passages)

            if teacher_scores is not None:
                if len(teacher_scores) > 0 and len(passages) > 0:
                    assert len(teacher_scores) == len(passages)

        return queries, passages, thought, teacher_scores


@dataclass
class DecoderOnlyEmbedderO1SameDatasetCollator(DataCollatorWithPadding):
    """
    EmbedCollator for SameDataset
    Note that after using this collator, the training_args should be set as:
        training_args.per_device_train_batch_size = 1
        training_args.dataloader_num_workers = 0    # avoid multi-processing

    Need to rewrite the load_collactor in absrunner
    """
    query_max_len: int = 32
    passage_max_len: int = 128
    example_query_max_len: int = 256
    sub_batch_size: int = -1
    # bos_token: str = '<s>'
    # eos_token: str = '</s>'
    # ins_token: str = '<instruct>'
    # query_bos_token: str = '<query>'
    # query_eos_token: str = '</query>'
    # resp_token = '<thought>'
    # emb_token = '<emb>'

    def __call__(self, features):
        # print(features[0])
        query = features[0][0]
        passage = features[0][1]
        thought = features[0][2]
        teacher_scores = features[0][3]
        no_in_batch_neg_flag = features[0][4]
        bs = len(query)

        if self.sub_batch_size is None or self.sub_batch_size <= 0:
            q_source_collated = self.tokenizer(
                query,
                padding=True,
                truncation=True,
                max_length=self.example_query_max_len, # the query have been truncated in __getitem__, max_length can be set to any value larger than prompt + query_max_len
                return_tensors="pt",
            ) # use for count the length of query
            d_collated = self.tokenizer(
                passage,
                padding=True,
                truncation=True,
                max_length=self.passage_max_len,
                return_tensors="pt",
            )
            # add <emb> token to the end of the passage
            # print(d_collated['input_ids'].shape)
            # print(bs)
            d_collated['input_ids'] = torch.cat((d_collated['input_ids'], torch.tensor([[self.tokenizer.convert_tokens_to_ids(EMB_TOKEN)]] * d_collated['input_ids'].shape[0])), dim=1)
            d_collated['attention_mask'] = torch.cat((d_collated['attention_mask'], torch.tensor([[1]] * d_collated['input_ids'].shape[0])), dim=1)
            # print(self.tokenizer.decode(d_collated['input_ids'][0]))

            q_len = q_source_collated['attention_mask'].sum(dim=1).tolist()
            examples = [q + q_l for q, q_l in zip(query, thought)]
            q_collated = self.tokenizer(
                examples, 
                padding=True, 
                truncation=True, 
                max_length=self.example_query_max_len, 
                return_tensors="pt"
            )
            # add <emb> token to the end of the query
            ## TODO:
            q_collated['input_ids'] = torch.cat((q_collated['input_ids'], torch.tensor([[self.tokenizer.convert_tokens_to_ids(EMB_TOKEN)]] * bs)), dim=1)
            q_collated['attention_mask'] = torch.cat((q_collated['attention_mask'], torch.tensor([[1]] * bs)), dim=1)
            # print(self.tokenizer.decode(q_collated['input_ids'][0]))

            # get tokenized query label, change q_collated_len part and the padding part to IGNORE_INDEX
            q_labels = q_collated['input_ids'].clone()
            q_lebels_len = q_collated['attention_mask'].sum(dim=1).tolist()
            assert self.tokenizer.padding_side == "left"
            response_id = self.tokenizer.convert_tokens_to_ids(RESPONSE_TOKEN)

            for idx,label in enumerate(q_labels):
                response_position = (label == response_id).nonzero(as_tuple=True)[0]
                assert len(response_position) < 2, f"multi <thought> in label: \n{features}"
                if len(response_position) == 0: # no <thought> in label
                    label[:] = IGNORE_INDEX
                elif len(response_position) == 1:
                    # print(self.tokenizer.decode(label[:response_position[0]+1]) + "\n[sep]\n" + self.tokenizer.decode(label[response_position[0]+1:]) + "\n")
                    label[:response_position[0]+1] = IGNORE_INDEX
                    label[-1] = IGNORE_INDEX
                    

                    
            q_collated['labels'] = q_labels
        else:
            raise NotImplementedError('sub_batch_size not supported')

        if isinstance(teacher_scores, list) and len(teacher_scores) == 0:
            teacher_scores = None

        return {
            "queries": q_collated,
            "passages": d_collated,
            "teacher_scores": teacher_scores,
            "no_in_batch_neg_flag": no_in_batch_neg_flag
        }