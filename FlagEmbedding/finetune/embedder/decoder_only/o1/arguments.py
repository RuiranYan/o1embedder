from typing import Optional, List
from dataclasses import dataclass, field

from FlagEmbedding.abc.finetune.embedder import (
    AbsEmbedderModelArguments,
    AbsEmbedderDataArguments,
    AbsEmbedderTrainingArguments,
)


def default_target_modules() -> List[int]:
    return ['v_proj', 'q_proj', 'k_proj', 'gate_proj', 'down_proj', 'o_proj', 'up_proj']


@dataclass
class DecoderOnlyEmbedderO1ModelArguments(AbsEmbedderModelArguments):
    peft_model_path: str = field(
        default='', metadata={"help": "The peft model checkpoint for initialization."}
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "If passed, will use LORA (low-rank parameter-efficient training) to train the model."}
    )
    lora_rank: int = field(
        default=64,
        metadata={"help": "The rank of lora."}
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": "The alpha parameter of lora."}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout rate of lora modules."}
    )
    target_modules: List[str] = field(
        default_factory=default_target_modules,
        metadata={"help": "The target modules to apply LORA."}
    )
    use_flash_attn: bool = field(
        default=False,
        metadata={"help": "If passed, will use flash attention to train the model."}
    )
    use_slow_tokenizer: bool = field(
        default=False,
        metadata={"help": "If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library)."}
    )
    from_peft: str = field(
        default=None
    )
    modules_to_save: str = field(
        default=None,
    )
    raw_peft: str = field(
        default=None
    )

    additional_special_tokens: Optional[str] = field(
        default=None,
        metadata={"help": "additional special tokens", "nargs": "+"}
    )

    save_merged_lora_model: bool = field(
        default=False,
        metadata={"help": "If passed, will merge the lora modules and save the entire model."}
    )


@dataclass
class DecoderOnlyEmbedderO1DataArguments(AbsEmbedderDataArguments):
    example_query_max_len: int = field(
        default=64,
        metadata={"help": "The max length of example query."}
    )
    # example_passage_max_len: int = field(
    #     default=96,
    #     metadata={"help": "The max length of example passage."}
    # )
    # retrieval_use_examples: bool = field(
    #     default=True,
    #     metadata={"help": "If passed, will use examples for retrieval."}
    # )
    # O1_suffix_str: str = field(
    #     default='\nResponse:',
    #     metadata={"help": "The suffix string for O1 dataset."}
    # )

@dataclass
class DecoderOnlyEmbedderO1TrainingArguments(AbsEmbedderTrainingArguments):
    sentence_pooling_method: str = field(default='cls', metadata={"help": "the pooling method. Available options: cls, mean, last_token. Default: cls", "choices": ['cls', 'mean', 'last_token', 'last_with_query', 'query_token']})
    loss_lambda: float = field(default=0.8, metadata={"help": "the lambda for the loss function."})