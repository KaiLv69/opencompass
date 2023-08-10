from opencompass.models import HuggingFaceCausalLM
from opencompass.models import LLMv3

models = [
    # dict(
    #     type=HuggingFaceCausalLM,
    #     abbr='llama-2-7b-hf',
    #     # path="meta-llama/Llama-2-7b-hf",
    #     # tokenizer_path='meta-llama/Llama-2-7b-hf',
    #     path="/mnt/petrelfs/chenkeyu1/models/llama2/Llama-2-7b-hf",
    #     tokenizer_kwargs=dict(padding_side='left',
    #                           truncation_side='left',
    #                           use_fast=False,
    #                           ),
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     model_kwargs=dict(device_map='auto'),
    #     batch_padding=False, # if false, inference with for-loop without batch padding
    #     run_cfg=dict(num_gpus=1, num_procs=1),
    # ),
    # dict(
    #     type=HuggingFaceCausalLM,
    #     abbr='llama-2-13b-hf',
    #     # path="meta-llama/Llama-2-13b-hf",
    #     # tokenizer_path='meta-llama/Llama-2-13b-hf',
    #     path="/mnt/petrelfs/chenkeyu1/models/llama2/Llama-2-13b-hf",
    #     tokenizer_kwargs=dict(padding_side='left',
    #                           truncation_side='left',
    #                           use_fast=False,
    #                           ),
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=8,
    #     model_kwargs=dict(device_map='auto'),
    #     batch_padding=False,  # if false, inference with for-loop without batch padding
    #     run_cfg=dict(num_gpus=2, num_procs=1),
    # )
    # dict(
    #     type=HuggingFaceCausalLM,
    #     abbr='llama-2-13b-hf',
    #     # path="meta-llama/Llama-2-7b-hf",
    #     # tokenizer_path='meta-llama/Llama-2-7b-hf',
    #     path="/mnt/petrelfs/chenkeyu1/models/llama2/Llama-2-13b-hf",
    #     tokenizer_kwargs=dict(padding_side='left',
    #                           truncation_side='left',
    #                           use_fast=False,
    #                           ),
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=8,
    #     model_kwargs=dict(device_map='auto'),
    #     batch_padding=False,  # if false, inference with for-loop without batch padding
    #     run_cfg=dict(num_gpus=2, num_procs=1),
    # ),

    # dict(
    #     type=HuggingFaceCausalLM,
    #     path="openlm-research/open_llama_3b_v2",
    #     tokenizer_kwargs=dict(padding_side='left',
    #                           truncation_side='left',
    #                           use_fast=False,
    #                           ),
    #     max_out_len=100,
    #     max_seq_len=2048,
    #     batch_size=16,
    #     model_kwargs=dict(device_map='auto'),
    #     batch_padding=False,  # if false, inference with for-loop without batch padding
    #     run_cfg=dict(num_gpus=2, num_procs=1),
    # ),

    dict(
        type=LLMv3,
        path="s3://model_weights/0331/linglongta_v4_2_cn_v3/9999/",
        tokenizer_path='/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model',
        tokenizer_type='v4',
        model_type="origin",
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1, num_procs=1)),
]
