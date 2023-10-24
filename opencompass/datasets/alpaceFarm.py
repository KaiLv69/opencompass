from alpaca_farm.auto_annotations import alpaca_leaderboard
from datasets import load_dataset

from opencompass.datasets import BaseDataset
from opencompass.openicl import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET


@LOAD_DATASET.register_module()
class AlpaceFarmDataset(BaseDataset):

    @staticmethod
    def load(path, name):
        dataset = load_dataset(path, name)['eval']

        def preprocess(example):
            content = example['input']
            if example['input'] != "":
                example['input'] = "### Input:\n" + content + "\n\n"
            return example

        dataset = dataset.map(preprocess)
        return dataset


@ICL_EVALUATORS.register_module()
class alpacaEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        dataset = load_dataset(path='tatsu-lab/alpaca_farm',
                               name='alpaca_farm_evaluation')['eval']
        instructions = dataset['instruction']
        inputs = dataset['input']
        outputs = []
        for pred, instruction, input in zip(predictions, instructions, inputs):
            outputs.append({
                'instruction': instruction,
                'input': input,
                'output': pred
            })
        df_results = alpaca_leaderboard(
            path_or_all_outputs=outputs,
            is_add_reference_methods=False,
            annotators_config='annotators/greedy_gpt4/configs.yaml',
        )
        # score = df_results.to_string(float_format="%.2f")
        score = df_results.iloc[0]['win_rate']
        score = round(score, 2)
        return {'score': score}