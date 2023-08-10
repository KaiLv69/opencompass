_path_prefix = "./data/BBH"
from os.path import exists
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import BBHDataset, BBHEvaluator, bbh_mcq_postprocess

bbh_multiple_choice_sets = [
    'temporal_sequences', 'disambiguation_qa', 'date_understanding', 'tracking_shuffled_objects_three_objects', 'penguins_in_a_table',
    'geometric_shapes', 'snarks', 'ruin_names', 'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_five_objects',
    'logical_deduction_three_objects', 'hyperbaton', 'logical_deduction_five_objects', 'logical_deduction_seven_objects', 'movie_recommendation',
    'salient_translation_error_detection', 'reasoning_about_colored_objects',
]
bbh_free_form_sets = [
    'multistep_arithmetic_two', 'navigate', 'dyck_languages', 'word_sorting', 'sports_understanding',
    'boolean_expressions', 'object_counting', 'formal_fallacies', 'causal_judgement', 'web_of_lies',
]
bbh_hint = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
bbh_reader_cfg = dict(input_columns=["input"], output_column="target")
bbh_datasets = []
for _name in bbh_multiple_choice_sets:
    _hint = open(f"{_path_prefix}/lib_prompt/{_name}.txt", 'r').read()
    lines = _hint.splitlines()
    question = lines[0].strip()
    question = question.replace("Task description: ","")
    hint = f"{bbh_hint}### Instruction:\n{question}\n\n### Input:"
    _hint = _hint.replace(question + '\n', "")
    _hint = _hint.replace("Q:", hint)
    _hint = _hint.replace("A: Let's think step by step.", "\n### Response: Let's think step by step.")
    _hint = _hint.replace("Task description: ","")
    _hint = _hint.strip()
    _hint = _hint+"\n\n"
    bbh_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            # template=dict(round=[dict(role="HUMAN", prompt=f"Follow the given examples and answer the question.\n{_hint}\n\nQ: </input>\nA: Let's think step by step.")]),
            template=f"{_hint}{bbh_hint}### Instruction:\n{question}\n\n### Input:\n{{input}}\n\n### Response: Let's think step by step.\n"
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer, max_out_len=512)
    )
    bbh_eval_cfg = dict(
        evaluator=dict(type=AccEvaluator),
        pred_postprocessor=dict(type=bbh_mcq_postprocess),
        dataset_postprocessor=dict(type=bbh_mcq_postprocess))

    bbh_datasets.append(
        dict(type=BBHDataset,
             path=f"{_path_prefix}/data",
             name=_name,
             abbr='bbh-' + _name,
             reader_cfg=bbh_reader_cfg,
             infer_cfg=bbh_infer_cfg.copy(),
             eval_cfg=bbh_eval_cfg.copy()))


for _name in bbh_free_form_sets:
    _hint = None
    if exists(f"{_path_prefix}/lib_prompt/{_name}.txt"):
        _hint = open(f"{_path_prefix}/lib_prompt/{_name}.txt", 'r').read()
        lines = _hint.splitlines()
        question = lines[0].strip()
        question = question.replace("Task description: ", "")
        hint = f"{bbh_hint}### Instruction:\n{question}\n\n### Input:"
        _hint = _hint.replace(question + '\n', "")
        _hint = _hint.replace("Q:", hint)
        _hint = _hint.replace("A: Let's think step by step.", "\n### Response: Let's think step by step.")
        _hint = _hint.replace("Task description: ", "")
        _hint = _hint.strip()
        _hint = _hint + "\n\n"
        bbh_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=f"{_hint}{bbh_hint}### Instruction:\n{question}\n\n### Input:\n{{input}}\n\n### Response: Let's think step by step.\n"
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer, max_out_len=512))
    else:
        bbh_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=f"{bbh_hint}### Instruction:\n{{input}}\n\n### Response: Let's think step by step.\n"
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer, max_out_len=512))
    bbh_eval_cfg = dict(evaluator=dict(type=BBHEvaluator), pred_role="BOT")

    bbh_datasets.append(
        dict(
            type=BBHDataset,
            path=f"{_path_prefix}/data",
            name=_name,
            abbr='bbh-' + _name,
            reader_cfg=bbh_reader_cfg,
            infer_cfg=bbh_infer_cfg.copy(),
            eval_cfg=bbh_eval_cfg.copy()))

del _name, _hint, _path_prefix
