from mmengine.config import read_base

from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner, SlurmRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import HuggingFaceCausalLM

with read_base():
    # import datasets that you want to eval
    from .datasets.alpaca_farm.alpaca_farm_gen_collie import alpaca_farm_datasets
    from .datasets.humaneval.humaneval_gen_collie import humaneval_datasets
    from .datasets.gsm8k.gsm8k_gen_collie import gsm8k_datasets
    from .datasets.mmlu.mmlu_gen_collie import mmlu_datasets
    from .datasets.bbh.bbh_gen_collie import bbh_datasets

    from .summarizers.groups.mmlu import mmlu_summary_groups
    from .summarizers.groups.bbh import bbh_summary_groups


datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='adamw',  # set abbr, which can be adamw adalomo lora ...
        path="huggyllama/llama-7b",  # your path to the model
        tokenizer_path="huggyllama/llama-7b",  # your path to the tokenizer
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False,
                              ),
        max_out_len=300,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto'),
        batch_padding=True,  # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=4, num_procs=1),  # num_gpus=1, 2, 4, 8 for 7b, 13b, 30b, 65b
    )
]

work_dir = './outputs/llama30b/'  # set work dir, which can be llama7b, llama13b, llama30b, llama65b

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=20000, gen_task_coef=10),
    # partitioner=dict(type='NaivePartitioner'),
    runner=dict(
        type=LocalRunner,
        max_num_workers=8,  # max gpus in your machine
        task=dict(type=OpenICLInferTask),
        # retry=5
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=8,
        task=dict(type=OpenICLEvalTask),
        # retry=0
    ),
)

summarizer = dict(
    dataset_abbrs=['mmlu', 'bbh', 'gsm8k_main', 'openai_humaneval', 'alpaca_farm'],
    summary_groups=sum([v for k, v in locals().items() if k.endswith("_summary_groups")], []),
)

# export OPENAI_API_KEY=your_api_key
# add -r to resume the previous run
# python run.py configs/eval_collie.py -r
