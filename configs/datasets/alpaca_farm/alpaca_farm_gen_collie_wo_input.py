from opencompass.datasets import alpacaEvaluator, AlpaceFarmDataset, AlpaceFarmDataset_no_input
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator

alpaca_farm_hint = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
alpaca_farm_reader_cfg = dict(input_columns=['instruction','input'], output_column="output")
alpaca_farm_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=f'{alpaca_farm_hint}### Instruction:\n{{instruction}}\n\n### Response:'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=300))

alpaca_farm_eval_cfg = dict(evaluator=dict(type=alpacaEvaluator))

alpaca_farm_no_input_datasets = [
    dict(
        abbr="alpaca_no_input",
        type=AlpaceFarmDataset_no_input,
        path='tatsu-lab/alpaca_farm',
        name='alpaca_farm_evaluation',
        reader_cfg = alpaca_farm_reader_cfg,
        infer_cfg=alpaca_farm_infer_cfg,
        eval_cfg=alpaca_farm_eval_cfg)
]
