from mmengine.config import read_base

from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner, SlurmRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    from .datasets.collections.leval2 import datasets
    from .my_model.my_collie_model import models

    # from .summarizers.medium import summarizer


work_dir = './outputs/2023_08_06_02/'

infer = dict(
    # partitioner=dict(type=SizePartitioner, max_task_size=20000, gen_task_coef=10),
    partitioner=dict(type='NaivePartitioner'),
    runner=dict(
        type=SlurmRunner,
        max_num_workers=10,
        task=dict(type=OpenICLInferTask),
        retry=3),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=SlurmRunner,
        max_num_workers=64,
        task=dict(type=OpenICLEvalTask),
        retry=4),
)

# python run.py configs/eval_collie2.py -p llm -r -l --debug 2>&1 | tee log.txt