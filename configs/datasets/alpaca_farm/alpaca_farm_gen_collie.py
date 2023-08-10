from opencompass.datasets import alpacaEvaluator, AlpaceFarmDataset
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator

alpaca_farm_hint_with_input = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
alpaca_farm_reader_cfg_with_input = dict(input_columns=['instruction','input'], output_column="output")
alpaca_farm_infer_cfg_with_input = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=f'{alpaca_farm_hint_with_input}### Instruction:\n{{instruction}}\n\n### Input:\n{{input}}\n\n### Response:'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=300))

alpaca_farm_eval_cfg_with_input = dict(evaluator=dict(type=alpacaEvaluator))

alpaca_farm_datasets_with_input = [
    dict(
        abbr="alpaca_with_input",
        type=AlpaceFarmDataset,
        path='tatsu-lab/alpaca_farm',
        name='alpaca_farm_evaluation',
        reader_cfg = alpaca_farm_reader_cfg_with_input,
        infer_cfg=alpaca_farm_infer_cfg_with_input,
        eval_cfg=alpaca_farm_eval_cfg_with_input)
]
