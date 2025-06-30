from transformers import HfArgumentParser

from FlagEmbedding.finetune.embedder.decoder_only.o1 import (
    DecoderOnlyEmbedderO1DataArguments,
    DecoderOnlyEmbedderO1TrainingArguments,
    DecoderOnlyEmbedderO1ModelArguments,
    DecoderOnlyEmbedderO1Runner,
)

parser = HfArgumentParser((
    DecoderOnlyEmbedderO1ModelArguments,
    DecoderOnlyEmbedderO1DataArguments,
    DecoderOnlyEmbedderO1TrainingArguments
))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()
model_args: DecoderOnlyEmbedderO1ModelArguments
data_args: DecoderOnlyEmbedderO1DataArguments
training_args: DecoderOnlyEmbedderO1TrainingArguments

runner = DecoderOnlyEmbedderO1Runner(
    model_args=model_args,
    data_args=data_args,
    training_args=training_args
)
runner.run()