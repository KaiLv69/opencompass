import os.path as osp
import tempfile
from typing import List

from datasets import load_dataset

from opencompass.datasets import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import (ICL_EVALUATORS, LOAD_DATASET,
                                  TEXT_POSTPROCESSORS)


@LOAD_DATASET.register_module()
class humanevalDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        import os
        os.environ[
            'http_proxy'] = 'http://chenkeyu1:Cky13291983702@10.1.8.50:33128/'
        os.environ[
            'https_proxy'] = 'http://chenkeyu1:Cky13291983702@10.1.8.50:33128/'
        os.environ[
            'HTTP_proxy'] = 'http://chenkeyu1:Cky13291983702@10.1.8.50:33128/'
        os.environ[
            'HTTPS_proxy'] = 'http://chenkeyu1:Cky13291983702@10.1.8.50:33128/'
        dataset = load_dataset(path)

        def split_string_by_token(text, token):
            index = text.find(token)
            if index == -1:
                # 如果特殊token不存在，则返回整个文本和空字符串
                return text, ''
            else:
                # 根据特殊token将文本分成两半
                part1 = text[:index]
                part2 = text[index + len(token):]
                return part1, part2

        def input_process(input):
            lines = input.splitlines()
            cleaned_lines = [line.lstrip() for line in lines]
            cleaned_text = '\n'.join(cleaned_lines)
            input = cleaned_text.strip()
            part1, part2 = split_string_by_token(input, '>>>')
            part1 = part1.replace('\n', ' ')
            input = part1 + '\n>>>' + part2

            return input

        def pre_process(example):
            prompt = example['prompt']
            start_index = prompt.find('"""')
            if start_index != -1:
                # 找到下一个连续三个双引号的位置
                end_index = prompt.find('"""', start_index + 3)

                if end_index != -1:
                    # 抽取出包含在连续三个双引号之间的部分
                    input = prompt[start_index + 3:end_index]
                    input = input_process(input)
                    example['input'] = input
                else:
                    print('未找到第二个连续三个双引号')
            else:
                start_index = prompt.find("'''")

                if start_index != -1:
                    # 找到下一个连续三个单引号的位置
                    end_index = prompt.find("'''", start_index + 3)

                    if end_index != -1:
                        # 抽取出包含在连续三个单引号之间的部分
                        input = prompt[start_index + 3:end_index]
                        input = input_process(input)
                        example['input'] = input
                    else:
                        print('未找到第二个连续三个单引号')
                else:
                    print('未找到第一个连续三个单引号')

            return example

        dataset = dataset.map(pre_process)
        return dataset


@ICL_EVALUATORS.register_module()
class HumanEvaluator(BaseEvaluator):
    """Evaluator for human eval."""

    def __init__(self, k: List[int] = [1, 10, 100]) -> None:
        try:
            from human_eval.data import HUMAN_EVAL, write_jsonl
            from human_eval.evaluation import evaluate_functional_correctness
            self.write_jsonl = write_jsonl
            self.HUMAN_EVAL = HUMAN_EVAL
            self.eval = evaluate_functional_correctness
        except ImportError:
            raise ImportError('Please install human_eval following'
                              'https://github.com/openai/human-eval/tree/'
                              'master#installation first.')
        self.k = k
        super().__init__()

    def score(self, predictions, references):

        predictions = [{
            'task_id': f'HumanEval/{i}',
            'completion': predictions[i]
        } for i in range(len(predictions))]
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = osp.join(tmp_dir, 'human_eval.json')
            self.write_jsonl(out_dir, predictions)
            score = self.eval(out_dir,
                              self.k,
                              n_workers=4,
                              timeout=3.0,
                              problem_file=self.HUMAN_EVAL)
            return {f'humaneval_{k}': score[k] * 100 for k in score}


@TEXT_POSTPROCESSORS.register_module('humaneval')
def humaneval_postprocess(text: str) -> str:
    text = text.split('\n\n')[0]
    if '```' in text:
        text = text.split('```')[1]
    if text.strip().startswith('def'):
        text = '\n'.join(text.split('\n')[1:])
    if not text.startswith('    '):
        if text.startswith(' '):
            text = '    ' + text.lstrip()
        else:
            text = '\n'.join(['    ' + line for line in text.split('\n')])
    return text
