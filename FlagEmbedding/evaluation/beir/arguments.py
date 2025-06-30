from dataclasses import dataclass, field

from FlagEmbedding.abc.evaluation import AbsEvalArgs, AbsEvalModelArgs


@dataclass
class BEIREvalArgs(AbsEvalArgs):
    use_special_instructions: bool = field(
        default=True, metadata={"help": "Whether to use specific instructions in `prompts.py` for evaluation. Default: False"}
    )

@dataclass
class BEIREvalModelArgs(AbsEvalModelArgs):
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    last_with_query: bool = field(
        default=False, metadata={"help": "whether to use last with query for embedding"}
    )
    n_ans: int = field(
        default=3, metadata={"help": "The number of responses"}
    )
