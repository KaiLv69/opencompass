# TODO: allow empty output-column
from opencompass.datasets import HumanEvaluator, humaneval_postprocess, humanevalDataset
from opencompass.openicl import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import SCInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator

humaneval_hint = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
humaneval_reader_cfg = dict(
    input_columns=['prompt','input'], output_column='task_id', train_split='test')
humaneval_infer_cfg = dict(prompt_template=dict(
                               type=PromptTemplate,
                               template=f'{humaneval_hint}### Instruction:\nComplete the following python code.\n\n### Input:\n{{input}}\n\n### Response:\n{{prompt}}',
                               # template='</E></prompt>',
                               ),
                           retriever=dict(type=ZeroRetriever),
                           inferencer=dict(type=GenInferencer, max_out_len=512))

humaneval_eval_cfg = dict(
    evaluator=dict(type=HumanEvaluator),
    k=[1, 10, 100],  # the parameter only for humaneval
    pred_postprocessor=dict(type=humaneval_postprocess),
)

humaneval_datasets = [
    dict(type=humanevalDataset,
         path='openai_humaneval',
         reader_cfg=humaneval_reader_cfg,
         infer_cfg=humaneval_infer_cfg,
         eval_cfg=humaneval_eval_cfg)
]
