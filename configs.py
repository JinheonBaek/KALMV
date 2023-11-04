from dataclasses import dataclass, field
from typing import List, Optional, Tuple

@dataclass
class GlobalArguments:
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for initialization"}
    )
    no_cuda: bool = field(
        default=False,
        metadata={"help": "Avoid using CUDA when available"}
    )
    debug: bool = field(
        default=False
    )
    stop: bool = field(
        default=False
    )
    exp_name: str = field(
        default='debug',
        metadata={"help": "Unique name of experiment"}
    )
    knowledge_base: str = field(
        default='kg',
    )
    edit_output: bool = field(
        default=False
    )
    num_edits: int = field(
        default=1,
    )

@dataclass
class DataArguments:
    data_name: str = field(
        default='WebQuestions',
        metadata={"help": "Name of dataset"}
    )
    data_path: str = field(
        default='./datasets',
        metadata={"help": "Path of dataset"}
    )
    data_type: str = field(
        default='KGQA'
    )
    batch_size: int = field(
        default=16,
        metadata={"help": "Batch size of dataset loaders"}
    )
    aliases: bool = field(
        default=True,
        metadata={"help": "Whether to consider aliases of answer entities"}
    )

@dataclass
class LanguageModelArguments:
    model_type: str = field(
        default='flan',
        metadata={"help": "Type of pretrained model"}
    )
    model_name_or_path: str = field(
        default='google/flan-t5-base',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    max_source_length: int = field(
        default=1024,
        metadata={"help": "The maximum total input sequence length"}
    )
    max_target_length: int = field(
        default=128,
        metadata={"help": "The maximum total sequence length for target text"}
    )
    cache_dir: Optional[str] = field(
        default='./cache',
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"}
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"}
    )
    device_map: str = field(
        default="auto"
    )
    question_prefix: str = field(
        default="Question: ",
        metadata={"help": "Text added prior to input question"}
    )
    question_postfix: str = field(
        default="Answer:",
        metadata={"help": "Text added next to input question"}
    )

@dataclass
class RetrieverArguments:
    use_retrieval: bool = field(
        default=True
    )
    retriever_name: str = field(
        default='mpnet',
        metadata={"help": "Name of retrieval model"}
    )
    retriever_top_k: int = field(
        default=10
    )
    retriever_batch_size: int = field(
        default=8192
    )
    retriever_sep_token: str = field(
        default=' '
    )
    index_dir: Optional[str] = field(
        default='./cache/pyserini'
    )

@dataclass
class VerifierArguments:
    use_verification: bool = field(
        default=True
    )
    verifier_name: str = field(
        default='google/flan-t5-base',
        metadata={"help": "Name of verifier model"}
    )
    verifier_generation_metric: str = field(
        default='accuracy',
        metadata={"help": "Name of verifier metric for generated answer"}
    )
    verifier_generation_threshold: float = field(
        default=0.5
    )
    verifier_retrieval_threshold: float = field(
        default=0.0
    )
    verifier_max_source_length: int = field(
        default=1024,
        metadata={"help": "The maximum total input sequence length"}
    )
    verifier_max_target_length: int = field(
        default=1,
        metadata={"help": "The maximum total sequence length for target text"}
    )
    fact_sep_token: str = field(
        default=' '
    )
    verifier_sample: bool = field(
        default=False
    )
    verifier_num_epochs: int = field(
        default=3
    )
    verifier_batch_size: int = field(
        default=32,
        metadata={"help": "Batch size of verifier dataset loaders"}
    )
    verifier_learning_rate: float = field(
        default=5e-5,
        metadata={"help": "Initial learning rate (after the potential warmup period) to use"}
    )
    verifier_weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay to use"}
    )
    ensemble: bool = field(
        default=False
    )
    verifier_num_instructions: int = field(
        default=5
    )
