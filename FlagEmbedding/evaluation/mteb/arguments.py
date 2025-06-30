from dataclasses import dataclass, field
from typing import List

from FlagEmbedding.abc.evaluation import AbsEvalArgs, AbsEvalModelArgs



@dataclass
class MTEBEvalArgs(AbsEvalArgs):
    languages: List[str] = field(
        default=None, metadata={"help": "Languages to evaluate. Default: eng"}
    )
    tasks: List[str] = field(
        default=None, metadata={"help": "Tasks to evaluate. Default: None"}
    )
    task_types: List[str] = field(
        default=None, metadata={"help": "The task types to evaluate. Default: None"}
    )
    use_special_instructions: bool = field(
        default=False, metadata={"help": "Whether to use specific instructions in `prompts.py` for evaluation. Default: False"}
    )
    use_special_examples: bool = field(
        default=False, metadata={"help": "Whether to use specific examples in `examples` for evaluation. Default: False"}
    )

@dataclass
class MTEBEvalModelArgs(AbsEvalModelArgs):
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    q2r_dic_path: str = field(
        default=None, metadata={"help": "The path to the query to response dictionary."}
    )
    last_with_query: bool = field(
        default=False, metadata={"help": "whether to use last with query for embedding"}
    )
    base_dpr: bool = field(
        default=False, metadata={"help": "whether to use base DPR model for embedding"}
    )
    version: str = field(
        default=None, metadata={"help": "The version of the model."}
    )
    n_ans: int = field(
        default=7, metadata={"help": "The number of responses"}
    )