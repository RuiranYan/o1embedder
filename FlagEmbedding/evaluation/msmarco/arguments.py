from FlagEmbedding.abc.evaluation import AbsEvalArgs, AbsEvalModelArgs

import os
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class MSMARCOEvalArgs(AbsEvalArgs):
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    

@dataclass
class MSMARCOEvalModelArgs(AbsEvalModelArgs):
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    last_with_query: bool = field(
        default=False, metadata={"help": "whether to use last with query for embedding"}
    )
    n_ans: int = field(
        default=3, metadata={"help": "The number of responses"}
    )